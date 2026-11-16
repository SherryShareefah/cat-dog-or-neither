# Cat, Dog, or Neither — Capstone Project

**Tech4Youth AI Engineering · Bamboni Inuwa**

The system looks at a folder of photographs of any size, shape or format and labels each one **dog**, **cat** or **neither**, with a confidence score.

## Files in this folder

| File | What it is |
|---|---|
| `Capstone.ipynb` | Training notebook: data loading, cleaning, both models, evaluation, saving |
| `Capstone_assessment.ipynb` | Assessment-day notebook (Colab): loads the saved model and predicts a folder in one cell |
| `cat_dog_neither.keras` | The trained model (17.1 MB) |
| `class_names.json` | Label order: 0 = dog, 1 = cat, 2 = neither |

**To run on new photos:** open `Capstone_assessment.ipynb` in Google Colab, run Cells 1 and 2, change the `FOLDER` path in Cell 3, and run it.

## Approach to "neither"

I chose **Approach A: three classes**, with "neither" trained as a real class.

I tested Approach B (threshold) first. A two-class cat-vs-dog model reached 99.5% validation accuracy, but it labelled all 8 sample "neither" photos as cat or dog. It called a yin-yang symbol **dog 93%**, a church **dog 89%** and a golf ball **dog 80%**. Softmax forces the scores to add up to 100%, so the model had no way to say "I don't know." Most of its "neither" guesses were "dog."

The confidence chart showed why a threshold struggles. Real pets clustered near 100% confidence, but "neither" photos were spread from 50% to 100%. Any "neither" photo near the top could not be caught without also rejecting real pets.

The three-class model, on the same 2,100 validation photos, caught **99.3%** of "neither" photos and wrongly rejected **0%** of real pets.

## Data

| Source | Use | Photos |
|---|---|---|
| Oxford-IIIT Pet | Cats and dogs (species label) | 2,371 cats, 4,978 dogs |
| Caltech-101 | "Neither" | 8,658 kept of 9,144 |
| Imagenette (320px) | "Neither", real cluttered scenes | 8,514 kept of 9,469 |

**Excluded from Caltech-101 (486 photos):** `dalmatian` (a dog breed), `snoopy` (cartoon dog), `garfield` (cartoon cat), `wild_cat`, `cougar_body`, `cougar_face` and `leopards` (cat family). Keeping them would teach the model that dogs and cats are "neither." Other animals (llama, panda, rooster and so on) stayed in on purpose, to teach that a non-pet animal is "neither."

**Excluded from Imagenette:** English springer (a dog breed).

**Why Imagenette was added:** most Caltech-101 photos show one object on a plain background. Alone, they risk teaching the model that "plain background = neither." Imagenette adds people, outdoor scenes and clutter.

**Final training set:** 2,492 dogs, 1,188 cats and 2,500 neither (6,180 photos). "Neither" was capped at 1,250 photos per source to keep the classes balanced.
**Validation set:** 1,029 dogs, 471 cats and 600 neither (2,100 photos), never used for training.

## Model

- **Pretrained EfficientNetB0** (ImageNet weights), frozen. Transfer learning reuses features learned from 1.2 million photos, so it trains in minutes on far fewer images than a model built from scratch.
- New head: global average pooling, dropout 0.3, then a 3-way softmax.
- **Augmentation:** flip, rotation, zoom, contrast and brightness, to mimic phone photos in different lighting.
- **Class weights** (cat ≈ 1.73, dog ≈ 0.83, neither ≈ 0.82), because the data had about 2.1 dogs for every cat.
- Images resized to 224×224 **with padding**, so nothing gets stretched. Phone rotation (EXIF), PNG transparency and greyscale are corrected before prediction. HEIC (iPhone) photos are supported.
- 6 epochs, Adam optimiser (learning rate 0.001).

## Results

**Validation set (2,100 photos): 99.4% accuracy, 12 mistakes**

| TRUE ↓ / PREDICTED → | dog | cat | neither |
|---|---|---|---|
| **dog** | 1,023 | 6 | 0 |
| **cat** | 2 | 469 | 0 |
| **neither** | 3 | 1 | 596 |

| Class | Precision | Recall |
|---|---|---|
| dog | 99.5% | 99.4% |
| cat | 98.5% | 99.6% |
| neither | 100% | 99.3% |

**Photos from outside the training data:** 6 real WhatsApp photos taken in Nigeria, none containing a cat or dog. All 6 were correctly labelled "neither" (6/6), with confidence between 59% and 98%.

## Where it fails

- **Small fluffy dogs** (Pomeranian, Chihuahua) get called **cat**. They're cat-sized with pointed ears.
- **Hairless Sphynx cats** get called **dog**. Without fur, the strongest "cat" signal is gone.
- **People on grass** can get called **dog**. Many dog photos are taken on grass, so the model partly learned the background.
- **Drawings and crowded scenes** (a rhino drawing, a sports field) confused it.
- **Every validation mistake had low confidence (55–79%).**
- **Real Nigerian photos:** on 6 WhatsApp photos, all were correctly called "neither," but confidence dropped to **59–98%**, compared with about 99% on validation. The model is less sure outside its training data.

## Limitations

- The validation "neither" photos come from the same sources as the training ones, so the 99.4% overstates real-world performance.
- The outside test is small: 6 photos, all "neither." The brief asks for at least 20 outside photos covering all three classes. I have not yet tested real local cat and dog photos, so that part of the system is unproven outside the downloaded data.
- No Nigerian photos were in the training set. Everyday Nigerian scenes (rooms, markets, okada, goats, chickens) are new to the model.
- A confidence score is the model's own estimate, not a probability of being right. The two-class model said "dog 93%" for a yin-yang symbol.
- The system gives one label per photo. A photo containing both a cat and a dog would need multi-label classification (sigmoid outputs and binary cross-entropy instead of softmax).
- A wolf or fox would probably be called "dog," because no wild canines were in the "neither" set.

## Improvements with more time

1. Build a held-out test set of 20+ of my own photos across all three classes, and report its confusion matrix.
2. Add 40+ of my own Nigerian "neither" photos to training. The pipeline already supports this (the `my_photos/neither` folder, with each photo repeated 10 times per epoch).
3. Add local cat and dog photos, especially indoor and low-light shots.
4. Fine-tune the top layers of EfficientNetB0 at a low learning rate.
5. Add wolves, foxes and other look-alike animals to "neither."
6. Combine Approach A with a confidence floor: answer "neither" or "not sure" when the top score is below a tested cut-off.

## Which mistake matters most

It depends on the use. For a system that alerts someone that a dog is loose on a property, calling an empty room "dog" is the worst mistake. False alarms make people switch the system off. That's why "neither" precision (100% on validation) matters most to me. For a pet photo-sorting app, a cat-dog swap is only an embarrassment.
