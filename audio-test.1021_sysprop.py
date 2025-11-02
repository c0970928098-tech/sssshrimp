import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModel ,LogitsProcessor
import torchaudio
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
from nltk.translate.meteor_score import meteor_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SAVE_ROOT = "/mnt/disk5/1021_new_test/"
PROJECTOR_SAVE_DIR = os.path.join(SAVE_ROOT, "projector")
LORA_SAVE_DIR = os.path.join(SAVE_ROOT, "lora_adapters")
CAPTION_SAVE_DIR = os.path.join(SAVE_ROOT, "captions")

class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = 3072
    max_length = 512

config = Config()

SYSTEM_PROMPT = (
                "You are Shrimp‑LLM, an expert AI for analyzing shrimp tank underwater audio. You receive an audio segment together with a user question. Tasks:\n1) QA description — provide clear, factual, concise analysis of audible events (e.g., shrimp walking, water and background sound)\n"
            )

class AllowListLogitsProcessor(LogitsProcessor):
    """只允許模型在 allowed_token_ids 裡選擇下一個 token。"""
    def __init__(self, allowed_token_ids):
        super().__init__()
        if isinstance(allowed_token_ids, torch.Tensor):
            self.allowed = allowed_token_ids.long()
        else:
            self.allowed = torch.tensor(allowed_token_ids).long()
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.allowed] = 0
        return scores + mask

class AudioProjector(nn.Module):
    def __init__(self, in_dim=512, out_dim=3072, dropout=0.1, gate_init=-2.0, l2norm=True):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.gate = nn.Parameter(torch.tensor(gate_init))  # α = sigmoid(gate)
        self.l2norm = l2norm

    def forward(self, clap_feat):  # [B, 512]
        x = self.ln(clap_feat)
        y = self.mlp(x)  # [B, out_dim]
        if self.l2norm:
            y = y / (y.norm(dim=-1, keepdim=True) + 1e-6)
        alpha = torch.sigmoid(self.gate)  # scalar ∈ (0,1)
        return y, alpha

class MultiModalInstructionTuningModel(nn.Module):
    def __init__(self, clap_model_name, llm_model_name, config, tokenizer, clap_ckpt_path=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # 先用 base 權重建立結構
        self.clap_model = AutoModel.from_pretrained(clap_model_name).to(config.device)

        # ✅ 若提供 finetune ckpt，就載入
        if clap_ckpt_path is not None:
            load_clap_weights_into_model(self.clap_model, clap_ckpt_path)

        # 再凍結
        for p in self.clap_model.parameters():
            p.requires_grad = False

        self.audio_projector = AudioProjector(in_dim=512, out_dim=config.hidden_size, dropout=0.1, gate_init=-2.0, l2norm=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(config.device)
        lora_config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            bias="none",
            target_modules=["self_attn.o_proj", "self_attn.qkv_proj"],
            task_type=TaskType.CAUSAL_LM
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.resize_token_embeddings(len(tokenizer))

    def extract_audio_features(self, waveform, sample_rate=48000):
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.size(1) < sample_rate:
            repeat = sample_rate // waveform.size(1) + 1
            waveform = torch.cat([waveform] * repeat, dim=1)

        if sample_rate != 48000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)(waveform)

        inputs = processor(audios=waveform.cpu().numpy(), sampling_rate=48000, return_tensors="pt").to(self.config.device)
        with torch.no_grad():
            features = self.clap_model.get_audio_features(**inputs).squeeze(0)
        return features

    def forward(self, audio_features, prompts, captions):
        device = audio_features.device  # ✅ 用 audio_features 的 device

        # 1) 準備 teacher-forcing 的輸入（prompt+caption），labels 只計 caption
        full_texts = [p + c for p, c in zip(prompts, captions)]
        tokenized = self.tokenizer(
            full_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=self.config.max_length
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        # 建議用「同一次 tokenize 的結果」估 prompt 長度，避免模板不一致
        # 這裡維持你的作法也可；若要更嚴謹，可另外對 prompts 做一次 tokenize 再取長度
        prompt_lens = [len(self.tokenizer(p, truncation=True, max_length=self.config.max_length)["input_ids"]) for p in prompts]

        # 2) Projector（LN+MLP+Gate）
        projected_audio, alpha = self.audio_projector(audio_features)  # [B, H], scalar

        # 3) 取得文字嵌入並在 <end_of_audio> 位置做「殘差注入」
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)   # [B, T, H]
        end_token_id = self.tokenizer.convert_tokens_to_ids("<end_of_audio>")

        B = input_ids.size(0)
        for b in range(B):
            pos = (input_ids[b] == end_token_id).nonzero(as_tuple=False)
            if len(pos) > 0:
                j = pos[0].item()
                inputs_embeds[b, j, :] = inputs_embeds[b, j, :] + alpha * projected_audio[b]

        # 4) 建 labels：遮掉 prompt 區段
        labels = input_ids.clone()
        for b, plen in enumerate(prompt_lens):
            labels[b, :plen] = -100

        # 5) 前向計算
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
    
def score_labels_by_logprob(model, tokenizer, inputs_embeds, attention_mask, candidate_texts, device):
    """
    對每個候選標籤（可能多 token）計算續寫的總 logprob，回傳 (best_label, scores_dict)
    重要：保留你已經做好的殘差注入的 inputs_embeds 作為前綴。
    """
    E = inputs_embeds  # [1, T, H]
    A = attention_mask # [1, T]
    scores = {}

    for lbl in candidate_texts:
        ids = tokenizer(lbl, add_special_tokens=False)["input_ids"]
        if len(ids) == 0:
            scores[lbl] = float("-inf"); continue

        # 逐 token 累積 logprob
        total_logp = 0.0
        cur_embeds = E
        cur_attn   = A

        for t in ids:
            # 取得下一步 logits
            with torch.no_grad():
                out = model.llm(inputs_embeds=cur_embeds, attention_mask=cur_attn, use_cache=False)
                # 只取最後一個位置的 logits
                next_logits = out.logits[:, -1, :]  # [1, V]
                logp = torch.log_softmax(next_logits, dim=-1)[0, t].item()
                total_logp += logp

            # 將選定 token 的嵌入接到序列末端，準備評分下一個 token
            tok_embed = model.llm.get_input_embeddings()(torch.tensor([[t]], device=device))  # [1,1,H]
            cur_embeds = torch.cat([cur_embeds, tok_embed], dim=1)
            cur_attn   = torch.cat([cur_attn, torch.ones_like(cur_attn[:, :1])], dim=1)

        scores[lbl] = total_logp

    # 取最高分標籤
    best = max(scores.items(), key=lambda x: x[1])[0]
    return best, scores


class AudioInstructionDataset(Dataset):
    """
    支援 JSONL 欄位:
      - audio_path (or file_name)
      - question (user prompt)
      - answer_text (for QA)
      - label (for classification)
      - system_prompt (可選, 覆蓋全域 SYSTEM_PROMPT)
    """
    def __init__(self, path, source_to_audio_dir, model, max_length=512):
        self.samples = []
        self.source_to_audio_dir = source_to_audio_dir or {}

        # 載入 JSONL
        with open(path, "r") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        def _normalize_label(s: str):
            if not s:
                return s
            s = s.strip().lower()
            mapping = {
                "water and background sound": "water",
                "shrimp walking": "walk",
                "walking": "walk",
                "walk": "walk",
            }
            return mapping.get(s, s)
        
        def _append_wav_if_needed(p: str) -> str:
            base, ext = os.path.splitext(p)
            if ext == "":
                return base + ".wav"
            return p
    
        for row in rows:
            raw_audio = row.get("audio_path", "").strip()
            raw_audio = _append_wav_if_needed(raw_audio)

            question = (row.get("question") or "").strip()
            task = (row.get("task") or "qa").strip().lower()
            answer_text = row.get("answer_text")
            label = row.get("label")
            label_set = row.get("label_set", None)   # ← 讀入 label_set
            source = row.get("meta", {}).get("source", "shrimp")

            # ---- 決定答案/標籤 ----
            if task == "cls":
                # 分類任務：只接受單一標籤
                label = _normalize_label(label)
                if not label:
                    print(f"[Skip] cls sample missing label: {raw_audio}")
                    continue
                caption = label  # 訓練目標就是單一標籤
                # 只有 cls 題保留 label_set（並正規化）；qa 題一律設 None
                if label_set:
                    label_set = [_normalize_label(x) for x in label_set]
                else:
                    label_set = None
            elif task == "qa":
                # QA 任務：只接受自然語句答案，不使用 label_set
                if not (answer_text and answer_text.strip()):
                    print(f"[Skip] qa sample missing answer_text: {raw_audio}")
                    continue
                caption = answer_text.strip()
                label_set = None  # 🔒 避免 QA 被受限解碼影響
            else:
                print(f"[Skip] unknown task '{task}': {raw_audio}")
                continue

            # ---- 決定音檔路徑（保持原邏輯）----
            if os.path.isabs(raw_audio):
                audio_path = raw_audio
            else:
                candidate = os.path.join(os.path.dirname(path), raw_audio)
                if os.path.exists(candidate):
                    audio_path = candidate
                else:
                    audio_dir = self.source_to_audio_dir.get(source, "")
                    audio_path = os.path.join(audio_dir, raw_audio)

            if not os.path.exists(audio_path):
                base, _ = os.path.splitext(audio_path)
                wav_try = base + ".wav"
                if os.path.exists(wav_try):
                    audio_path = wav_try
                else:
                    print(f"[Skip] Missing audio: {audio_path}")
                    continue

            # ---- 建立對話樣本 ----
            system_prompt = SYSTEM_PROMPT
            user_prompt = "<end_of_audio>\n" + question

            self.samples.append({
                "audio_path": audio_path,
                "source": source,
                "task": task,            # ★ 存 task
                "label_set": label_set,  # ★ 只有 cls 才會有；qa 是 None
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": caption + "<|end|>\n"}
                ]
            })
        print(f"✅ Total valid samples: {len(self.samples)}  (from {path})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        task = sample["task"]
        label_set = sample.get("label_set", None)  # ★ 新增
        audio_path = sample["audio_path"]
        source = sample["source"]
        messages = sample["messages"]

        # prompt 到 <|assistant|> 前
        prompt = model.tokenizer.apply_chat_template(
            messages[:2],
            add_generation_prompt=True,
            tokenize=False
        )
        caption = messages[2]["content"]

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            features = model.extract_audio_features(waveform, sample_rate)
            return features, prompt, caption, source, label_set, task
        except Exception as e:
            print(f"[LoadError] {audio_path}: {e}")
            return None, None, None, None


def load_clap_weights_into_model(clap_model: torch.nn.Module, ckpt_path: str):
    """
    將本地 finetuned CLAP 權重載入到 AutoModel 建立的 clap_model。
    兼容幾種常見格式：
    - 純 state_dict
    - PyTorch Lightning: {"state_dict": ...}
    - DataParallel: key 以 "module." 開頭
    會自動做 key 正規化並只載入可對上的權重（strict=False）。
    """
    if not os.path.exists(ckpt_path):
        print(f"[CLAP-Load] ❌ ckpt not found: {ckpt_path}")
        return

    print(f"[CLAP-Load] Loading CLAP checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 取出 state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        print("[CLAP-Load] ❌ Unsupported checkpoint format.")
        return

    # 可能的前綴需要去掉：module., model., clap_model., clap., audio_branch. 等
    def normalize_key(k: str) -> str:
        for prefix in ["module.", "model.", "clap_model.", "clap.", "audio_backbone.", "audio_branch."]:
            if k.startswith(prefix):
                return k[len(prefix):]
        return k

    sd = {normalize_key(k): v for k, v in sd.items()}

    # 只保留 clap_model 中存在的 key
    model_sd = clap_model.state_dict()
    filtered = {}
    matched, skipped = 0, 0
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
            matched += 1
        else:
            skipped += 1

    missing, unexpected = clap_model.load_state_dict(filtered, strict=False)
    print(f"[CLAP-Load] matched: {matched}, skipped(by shape/key): {skipped}")
    if missing:
        print(f"[CLAP-Load] missing keys: {list(missing)[:10]} ... (total {len(missing)})")
    if unexpected:
        print(f"[CLAP-Load] unexpected keys: {list(unexpected)[:10]} ... (total {len(unexpected)})")
    print("[CLAP-Load] ✅ done.")

def collate_fn(batch):
    batch = [s for s in batch if s[0] is not None]
    if len(batch) == 0:
        return None
    audio_features = torch.stack([x[0] for x in batch])
    prompts  = [x[1] for x in batch]
    captions = [x[2] for x in batch]
    sources  = [x[3] for x in batch]
    label_sets = [x[4] for x in batch]
    tasks = [x[5] for x in batch]                 # ✨
    return audio_features, prompts, captions, sources, label_sets, tasks


def save_lora_adapter(model, epoch):
    save_path = os.path.join(LORA_SAVE_DIR, f"epoch_{epoch}/")
    os.makedirs(save_path, exist_ok=True)
    if hasattr(model.llm, "save_pretrained"):
        model.llm.save_pretrained(save_path)
        print(f"✅ LoRA adapter saved to '{save_path}'")


def save_captions(captions, sources, epoch):
    os.makedirs(CAPTION_SAVE_DIR, exist_ok=True)
    shrimp_data = []
    for cap, src in zip(captions, sources):
        item = {"caption": cap, "source": src}
        shrimp_data.append(item)
    with open(os.path.join(CAPTION_SAVE_DIR, f"shrimp_epoch_{epoch}.jsonl"), "w") as f:
        for item in shrimp_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
import re

def _simple_tokenize(text: str):
    # 不依賴 nltk 資料檔，避免環境缺少 punkt。
    # 將字母數字序列與單一符號分開：如 ["A", "faint", "rustling", ",", "sound", "..."]
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

def run_validation(model, dataloader, tokenizer, config, epoch):
    import re
    model.eval()
    total_meteor = 0.0
    meteor_count = 0

    # 統計分類的 micro-F1（= 正確率）
    cls_correct = 0
    cls_total = 0

    for batch in dataloader:
        if batch is None: 
            continue
        audio_features, prompts, captions, _, label_sets, tasks = batch
        audio_features = audio_features.to(config.device)

        for i in range(len(prompts)):
            task = tasks[i]
            label_set = label_sets[i] or []
            prompt  = prompts[i:i+1]
            caption = captions[i] or ""

            # 1) tokenize
            tokenized = tokenizer(
                prompt, return_tensors="pt",
                padding=True, truncation=True, max_length=config.max_length
            )
            input_ids = tokenized["input_ids"].to(config.device)
            attention_mask = tokenized["attention_mask"].to(config.device)
            if input_ids.size(1) == 0 or attention_mask.size(1) == 0:
                print("⚠️ Warning: Empty input_ids/attention_mask; skip.")
                continue

            # 2) 殘差注入
            inputs_embeds = model.llm.get_input_embeddings()(input_ids)  # [1, T, H]
            end_token_id  = tokenizer.convert_tokens_to_ids("<end_of_audio>")
            projected, alpha = model.audio_projector(audio_features[i:i+1])  # [1, H], scalar
            pos = (input_ids[0] == end_token_id).nonzero(as_tuple=False)
            if len(pos) > 0:
                j = pos[0].item()
                inputs_embeds[0, j, :] = inputs_embeds[0, j, :] + alpha * projected[0]

            # 3) 產生 / 打分
            if task == "cls" and len(label_set) > 0:
                # 分類：用 logprob 計分選最大者
                candidate_texts = label_set
                pred, _ = score_labels_by_logprob(
                    model, tokenizer, inputs_embeds, attention_mask, candidate_texts, config.device
                )
            else:
                # QA：自由生成
                with torch.no_grad():
                    out = model.llm.generate(
                        input_ids=input_ids,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=64,
                        do_sample=False,
                        return_dict_in_generate=True
                    )
                    generated_ids = out.sequences
                new_len = inputs_embeds.shape[1]
                new_tokens = generated_ids[:, new_len:]
                preds = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                pred = (preds[0].strip() if len(preds) > 0 else "")

            # 正規化文字
            def _clean_text(s: str) -> str:
                s = re.sub(r"<\|.*?\|>", "", s)
                s = s.replace("<end_of_audio>", "")
                s = re.sub(r"\s+", " ", s).strip()
                return s

            q_raw = prompts[i]
            m = re.search(r"<\|user\|>\s*(.*?)\s*<\|assistant\|>", q_raw, flags=re.DOTALL)
            q_show = _clean_text(m.group(1) if m else q_raw)

            ref_show  = _clean_text(caption)
            pred_show = _clean_text(pred)

            # 顯示 + 計分
            print(f"📝 QUESTION: {q_show}")
            print(f"🔹REF: {ref_show}")
            print(f"🔸PRED: {pred_show}")

            if task == "cls":
                # 逐筆 F1（單標籤 → 完全相同=1，否則=0）
                ref_norm  = ref_show.lower()
                pred_norm = pred_show.lower()
                f1_inst = 1.0 if pred_norm == ref_norm and ref_norm != "" else 0.0
                print(f"🎯 F1score: {f1_inst:.4f}\n")

                cls_total += 1
                cls_correct += int(f1_inst == 1.0)
            else:
                # QA 用 METEOR
                pred_norm = re.sub(r"\s+", " ", pred_show.lower())
                ref_norm  = re.sub(r"\s+", " ", ref_show.lower())

                pred_toks = _simple_tokenize(pred_norm)
                ref_toks  = _simple_tokenize(ref_norm)

                score = meteor_score([ref_toks], pred_toks)
                print(f"🎯 METEOR: {score:.4f}\n")

                total_meteor += float(score)
                meteor_count += 1

    avg_meteor = (total_meteor / meteor_count) if meteor_count > 0 else 0.0
    epoch_f1 = (cls_correct / cls_total) if cls_total > 0 else 0.0

    # 期末彙總列印（兩個指標各自只在有資料時顯示）
    tail = []
    tail.append(f"🧪 METEOR 評分 @ epoch {epoch}: {avg_meteor:.4f}")
    tail.append(f"F1score 評分 @ epoch {epoch}: {epoch_f1:.4f}")
    print("  ".join(tail))

    model.train()
    return avg_meteor  # 仍回傳 METEOR（若需要也可改成回傳 (avg_meteor, epoch_f1)）



def train_instruction_tuning(train_path, val_path, source_to_audio_dir, clap_model_name, llm_model_name,
                             processor, config, tokenizer, batch_size, epochs):
    os.makedirs(PROJECTOR_SAVE_DIR, exist_ok=True)
    global model
    model = MultiModalInstructionTuningModel(clap_model_name, llm_model_name, config, tokenizer).to(config.device)

    # 這邊載入finetune 過的 CLAP 權重
    CLAP_CKPT_PATH = "/mnt/disk6/clap_finetune/shrimp_full/best.ckpt"
    load_clap_weights_into_model(model.clap_model, CLAP_CKPT_PATH)

    # 這邊凍結 CLAP 權重
    for p in model.clap_model.parameters():
        p.requires_grad = False

    # 這邊載入訓練與驗證資料
    train_dataset = AudioInstructionDataset(train_path, source_to_audio_dir, model, max_length=config.max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = AudioInstructionDataset(val_path, source_to_audio_dir, model, max_length=config.max_length)
    val_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        all_captions, all_sources = [], []
        print(f"\n🔁 Epoch {epoch+1}/{epochs}")
        for batch in train_loader:
            if batch is None: continue
            audio_features, prompts, captions, sources, label_sets, tasks = batch  # ✨
            audio_features = audio_features.to(config.device)
            outputs = model(audio_features, prompts, captions)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_captions.extend(captions)
            all_sources.extend(sources)
        print(f"📉 Loss: {total_loss / max(1,len(train_loader)):.4f}")
        run_validation(model, val_loader, tokenizer, config, epoch+1)
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(PROJECTOR_SAVE_DIR, f"projector_epoch_{epoch+1}.pth"))
            save_lora_adapter(model, epoch+1)
        save_captions(all_captions, all_sources, epoch+1)

# 主程式碼
if __name__ == "__main__":
    TRAIN_PATH = "/mnt/disk5/shrimp_walk/train.split.50.jsonl"
    VAL_PATH   = "/mnt/disk5/shrimp_walk/val.sys.enq.jsonl"
    SOURCE_TO_AUDIO_DIR = {"audio_path": "/mnt/disk5/shrimp_walk/wav_output/", "user_csv": "/mnt/disk5/shrimp_walk/wav_output/"}
    CLAP_MODEL_NAME = "laion/clap-htsat-unfused"
    LLM_MODEL_NAME  = "microsoft/Phi-3.5-mini-instruct"
    processor = AutoProcessor.from_pretrained(CLAP_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<end_of_audio>"]})
    train_instruction_tuning(
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        source_to_audio_dir=SOURCE_TO_AUDIO_DIR,
        clap_model_name=CLAP_MODEL_NAME,
        llm_model_name=LLM_MODEL_NAME,
        processor=processor,
        config=config,
        tokenizer=tokenizer,
        batch_size=8,
        epochs=100
    )
