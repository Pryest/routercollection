from calibration_model import CalibrationDecoder
from transformers import AutoTokenizer
from trainer import Trainer


model_path = "/cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549/"
train_data_path = "/cpfs01/shared/llm_ddd/b_cpfs_trasfer_data/pengrunyu/1205/workdir/12-19-10:39:08/llama-7b/"
eval_data_path = "/cpfs01/shared/llm_ddd/b_cpfs_trasfer_data/pengrunyu/1205/workdir/12-19-10:45:00/llama-7b/"


save_path = "/cpfs01/shared/llm_ddd/b_cpfs_trasfer_data/pengrunyu/ckpts/llama-7b-triviaqa-hard/"

epochs = 200

train_batch_size = 7680 // 8
max_train_batch_size = 32
micro_num = 1

max_eval_batch_size = 32


model = CalibrationDecoder(model_path)
model.model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    save_path=save_path,
    train_data_folder=train_data_path,
    eval_data_folder=eval_data_path,
    train_batch_size=max_train_batch_size,
    eval_batch_size=max_eval_batch_size,
    micro_num=micro_num,
    max_epochs=epochs,
)

trainer.train()
