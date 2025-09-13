---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:3037
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: Checker verifies quantity, quality, condition, value and type of
    material, product, purchased, sold or produced against records, reports or specifications.
    Examines articles for defects in making or flows in material and returns defective
    work for correction or as rejects; checks packages, bottles etc., for proper markings
    and labels according to instructions; checks and examines incoming and outgoing
    goods and materials against lists and delivery notes for quality, quantity or
    number and reports or records defects or deficiencies. May sort data or items
    into predetermined sequence, groups, quantity. May record items verified. May
    be designated according to material handled, industry in which employed or type
    of establishment worked as CURRENCY CHECKER; CARGO CHECKER; WAREHOUSE CHECKER;
    CHECKER, POTTERY WARE; CHECKER, GARMENTS.
  sentences:
  - Terminal Manager
  - Hut Maker
  - Checker
- source_sentence: 'PUC Attendant needs to assist customers’ in checking the vehicles
    and understanding the compliance pertaining to PUC. Qualification Pack Details:
    QP NOS Reference ASC/Q9601 QP NOS Name PUC Attendant Level 2 NSQF Level 2'
  sentences:
  - Painters and Related Workers, Other
  - PUC Attendant
  - Oil Crusher Operator, Animal Driven
- source_sentence: Instrumental Musician Percussion Instrument plays musical percussion
    instruments such as Tabla, Drum, Pakhawaj etc. by hands or sticks usually to provide
    rhythmic company to other musicians. Tunes instrument to required pitch by tightening
    or loosening cords holding leather pieces fixed atop or at both ends of instrument.
    Plays instrument gently with hand or stick or both to provide rhythm, alone or
    in accompaniment to other musical instrument(s). May sing classical or light songs
    while playing instrument. May be known as TABLA PLAYER, DRUMMER, MRIDANGlST, PAKHAWAJ
    PLAYER, CYMBAL1ST, etc. according to instrument played.
  sentences:
  - Electrical Fitter
  - Instrumental Musician, Percussion Instrument
  - Four Cutter
- source_sentence: 'Press Shop Supervisor is responsible for supervising the metal
    pressing and sheeting activities to create well-formed sheet metal components
    for automobile frames and auto components using manual, hydraulic or pneumatic
    presses, maintaining process parameters, conducting quality checks on output product,
    deploying manpower as per requirement, guiding operatives and technicians to complete
    the assigned task, maintaining a safe and healthy working environment on the shop
    floor and maintaining records related to production, rejections, material movement
    and manpower productivity for a line/shift Qualification Pack Details: QP NOS
    Reference ASC/Q3404 QP NOS Name Press Shop Supervisor NSQF Level 5 ISCO 08 Unit
    Group Details : Code 3122 Title Manufacturing Supervisors'
  sentences:
  - Customer Relation Executive
  - Press Shop Supervisor
  - Warp Knitter/Knitting Machine Operator-Warp Knitter
- source_sentence: 'General Manager, Recreation and Entertainment controls, co-ordinates
    and supervises, within authority delegated, efficient and economic utilisation
    of men, money and material in public and private organizations, establishments
    etc., or one or more of its branches or departments, engaged in producing, distributing
    and exhibiting motion pictures; producing and presenting stage and circus shows;
    organizing and presenting radio and television broadcasts; operating carnivals
    and amusement parks; organizing games, hunting, fishing, excursions, competitions,
    etc., and providing other entertainment and recreation services. Included are:
    MANAGER, CINEMA. MANAGER, STUDIO. MANAGER, THEATRE. MANAGER, STAGE. MANAGER, CIRCUS.
    MANAGER, VARIETY SHOW. MANAGER, AMUSEMENT PARK, MANAGER, MAGIC SHOW, MANAGER,
    TOURNAMENT, MANAGER, STADIUM, MANAGER, CLUB, MANAGER, SWIMMING POOL, MANAGER,
    DANCE HALL, MANAGER, ORCHESTRA, MANAGER, EXHIBITION, MANAGER, CARNIVAL, MANAGER,
    RACE COURSE, MANAGER, GYMNASIUM.'
  sentences:
  - General Manager, Recreation and Entertainment
  - Dyed-Yarn Operator (Textile)
  - Education Methods Specialists, Other
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained on the csv dataset. It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - csv
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'General Manager, Recreation and Entertainment controls, co-ordinates and supervises, within authority delegated, efficient and economic utilisation of men, money and material in public and private organizations, establishments etc., or one or more of its branches or departments, engaged in producing, distributing and exhibiting motion pictures; producing and presenting stage and circus shows; organizing and presenting radio and television broadcasts; operating carnivals and amusement parks; organizing games, hunting, fishing, excursions, competitions, etc., and providing other entertainment and recreation services. Included are: MANAGER, CINEMA. MANAGER, STUDIO. MANAGER, THEATRE. MANAGER, STAGE. MANAGER, CIRCUS. MANAGER, VARIETY SHOW. MANAGER, AMUSEMENT PARK, MANAGER, MAGIC SHOW, MANAGER, TOURNAMENT, MANAGER, STADIUM, MANAGER, CLUB, MANAGER, SWIMMING POOL, MANAGER, DANCE HALL, MANAGER, ORCHESTRA, MANAGER, EXHIBITION, MANAGER, CARNIVAL, MANAGER, RACE COURSE, MANAGER, GYMNASIUM.',
    'General Manager, Recreation and Entertainment',
    'Dyed-Yarn Operator (Textile)',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.9258, -0.1114],
#         [ 0.9258,  1.0000, -0.1289],
#         [-0.1114, -0.1289,  1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### csv

* Dataset: csv
* Size: 3,037 training samples
* Columns: <code>Unit_Description</code> and <code>Unit_Title</code>
* Approximate statistics based on the first 1000 samples:
  |         | Unit_Description                                                                     | Unit_Title                                                                       |
  |:--------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
  | type    | string                                                                               | string                                                                           |
  | details | <ul><li>min: 11 tokens</li><li>mean: 130.63 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 7.06 tokens</li><li>max: 30 tokens</li></ul> |
* Samples:
  | Unit_Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Unit_Title                                                    |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | <code>Actuary calculates and fixes premium rates for insurance against different types of risks on the basis of statistical, mathematical and financial calculations involving probability of future payment or contingencies. Collects and analyses data from different sources concerning losses to person from death, disability, sickness, injury etc. and losses to property from fire, burglary, explosion, hazards etc. Considers probable frequency of such risks, calculates their costs, and fixes rates of premiums for different types of risks taking in to account money market and economic conditions and probable future trends thereon. Studies continually about new developments, business trends, legislative, social and other factors affecting insurance business. Prepares contract provisions of insurance and pension plans. Determines proper basis and methods of evaluating liability of insurance and pension organizations. Recommends to management suitable measures regarding future policies and course of...</code> | <code>Actuary</code>                                          |
  | <code>Galvanizer applies coating of zinc on ferrous articles by dipping them in molten zinc. Checks and controls quantity, quality and temperature of acid (hydrochloric  Divison 8 acid), flux (zinc chloride) and zinc baths. Preheat articles if necessary and dips or passes them either manually or mechanically through, acid, water, flux and zinc baths successively at controlled speed. Skims dirt from baths and continues operation with necessary adjustment of solution, temperature etc., ensuring regular and uniform coating. May similarly apply tin coating using palm oil as flux and be designated as TIN PLATER or TINNING MACHINE OPERATOR. May regulate temperature by gauges and by colour of melting metals.</code>                                                                                                                                                                                                                                                                                                            | <code>Galvanizer/Operator- Electroplating, Galvanising</code> |
  | <code>Spun Pipe Machine Operator operates spinning machine to make spun metal pipes. Moves pipe spinning machine under spout of ladle containing molten metal and allows metal to flow into vertical pipe mould; moves machine back to normal position and starts mould spinning rapidly to evenly distribute metal and eliminate gas bubbles. Allows metal to set and removes pipe from mould, using pipe putter. Gets mould filled with required quantity of molten metal.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | <code>Spun Pipe Machine Operator</code>                       |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Evaluation Dataset

#### csv

* Dataset: csv
* Size: 338 evaluation samples
* Columns: <code>Unit_Description</code> and <code>Unit_Title</code>
* Approximate statistics based on the first 338 samples:
  |         | Unit_Description                                                                     | Unit_Title                                                                       |
  |:--------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
  | type    | string                                                                               | string                                                                           |
  | details | <ul><li>min: 14 tokens</li><li>mean: 132.85 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 7.14 tokens</li><li>max: 21 tokens</li></ul> |
* Samples:
  | Unit_Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Unit_Title                             |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------|
  | <code>Sales Consultant (Retail) Level 5 handles potential customer leads, presents value proposition for vehicles and manages vehicle retail sales. Qualification Pack Details: QP NOS Reference ASC/Q1005 QP NOS Name Sales Consultant (Retail) Level 5 NSQF Level 5</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | <code>Sales Consultant (Retail)</code> |
  | <code>Calibration Engineer tests and standardizes medical devices as per device manufacturer’s specification or customer requirement. The individual at work compares the measurements of medical device with the master equipment and then certifies medical device as standardized or non-standardized as per device manufacturer’s specification or customer’s requirement. Qualification Pack Details: QP NOS Reference ELE/Q8002 QP NOS Name Calibration Engineer NSQF Level 4</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | <code>Calibration Engineer</code>      |
  | <code>Examiner, Wood Working; Viewer Wood Working; Wood Grader examines wood and finished wooden articles on completion or at different stages of production and ensures adherence to prescribed tolerances and specifications. Studies drawings and specifications and notes types of articles to be made. Examines type and quality of wood for grains, knots etc. for grading and to ensure adherence to prescribed specifications. Checks marking, sawing, shaping, joining, setting etc. along with drawing or sample using foot rule, callipers and other instruments at various stages of production and ensures required finish of completed article. Records defects where noticed, rejects defective pieces or suggests rectifications, if possible. May check machine set up, blending of patches, inlaying, seasoning of wood and like factors for manufacture of particular items such as sports goods, cabinets, furniture, rifle butts etc. May estimate material and labour cost.</code> | <code>Examiner, Wood Working</code>    |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 6

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 6
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 1.3158 | 500  | 0.0002        |
| 2.6316 | 1000 | 0.0009        |
| 3.9474 | 1500 | 0.0004        |
| 5.2632 | 2000 | 0.0001        |


### Framework Versions
- Python: 3.12.3
- Sentence Transformers: 5.1.0
- Transformers: 4.52.4
- PyTorch: 2.7.1+cu126
- Accelerate: 1.10.1
- Datasets: 4.0.0
- Tokenizers: 0.21.4

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->