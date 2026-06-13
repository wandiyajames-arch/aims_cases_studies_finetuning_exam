# Data Folder

This folder intentionally contains only one starter dataset:

```text
syntetic_wolof_instruct_data.jsonl
```

Students must build a clean data pipeline that creates separate source files,
for example:

```text
data/wolof_aya.jsonl
data/wolof_soynade.jsonl
data/wolof_synth.jsonl
```

Then they must create chat-format files:

```text
data/chat_aya.json
data/chat_soynade.json
data/chat_synth.json
```

Then split the data:

```text
data/splits/*_train.json
data/splits/*_validation.json
data/splits/*_eval.json
data/splits/eval_all.jsonl
```

Generated chat files and splits are ignored by Git by default. Students should
include their methodology and final counts in the report and model card.
