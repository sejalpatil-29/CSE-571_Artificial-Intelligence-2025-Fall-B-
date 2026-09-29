# CSE 571 – Neural Network for Collision Prediction

A small simulated robot (pygame + pymunk) wanders a walled arena. I collect data
on what happens when it takes an action, train a PyTorch network to **predict
whether an action will cause a collision**, then use that network so the robot
can seek goals while only choosing actions it thinks are safe.

---

## How it works (the big picture)

```
Part 1: collect_data.py     -> robot wanders, records samples   -> training_data.csv
Part 2: Data_Loaders.py     -> loads + normalizes CSV, 80/20 split into DataLoaders
Part 3: Networks.py         -> the neural network + evaluate()
Part 4: train_model.py      -> trains, saves saved_model.pkl + scaler.pkl
        goal_seeking.py     -> demo: robot seeks 10 goals using the trained model
```

**One sample = one action = one CSV row, 7 columns, no header:**

| Columns | Meaning |
|---|---|
| 1-5 | Five distance-sensor readings (sensor range 150) |
| 6 | Action, an integer from -5 to 5 (turn = action x 0.1 pi from current heading) |
| 7 | Collision: 0 = none, 1 = collided |

The network input is columns 1-6 (5 sensors + action); the label is column 7.

---

## Repo layout

| Path | What it is |
|---|---|
| `assignment_part1/` | Data collection. `collect_data.py` (my code), `SimulationEnvironment.py`, `SteeringBehaviors.py` (Wander), `Helper.py`, `assets/` (robot images), `saved/` (data output + provided sample CSV) |
| `assignment_part2/` | `Data_Loaders.py` only (Dataset + DataLoaders) |
| `ass3/` | Part 3: `Data_Loaders.py`, `Networks.py`, `saved/` |
| `assignment_part4/` | Final, complete pipeline: `Data_Loaders.py`, `Networks.py`, `train_model.py`, `goal_seeking.py`, `SimulationEnvironment.py`, `SteeringBehaviors.py` (Wander + Seek), `Helper.py`, `assets/`, `saved/` |
| `AI commands.txt` | My setup commands (PowerShell versions) |
| `CSE 571_...Overview Document.pdf` | Assignment spec |
| `.gitignore` | Keeps `venv/` etc. out of Git |

**`assignment_part4/` is the one to use.** Parts 1-3 are earlier stages of the same code.

Files marked "STUDENTS DO NOT EDIT" in the starter code (`SimulationEnvironment.py`,
`SteeringBehaviors.py`) are the provided simulator. Files marked "STUDENTS EDIT"
are the ones I wrote.

---

## Requirements

- Python **3.12**
- Pinned packages:

```
cython==3.0.3  matplotlib==3.9.3  scikit-learn==1.5.2  scipy==1.14.1
pymunk==6.9.0  pygame==2.6.1  pillow==11.0.0  numpy==2.1.3  noise==1.2.2
torch==2.5.1   torchvision==0.20.1
```

A screen is needed (pygame opens a window).

## Setup

**Git Bash (Windows):**

```bash
python -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
pip install cython==3.0.3 matplotlib==3.9.3 scikit-learn==1.5.2 scipy==1.14.1 pymunk==6.9.0 pygame==2.6.1 pillow==11.0.0 numpy==2.1.3 noise==1.2.2 torch==2.5.1 torchvision==0.20.1
```

**PowerShell:** use `py -3.12 -m venv venv` and `.\venv\Scripts\Activate.ps1`
instead. (`py` and `.ps1` do not work in Git Bash.)

`venv/` is git-ignored. Never commit it.

---

## Run the full pipeline

From the repo root, with the venv active:

```bash
# 1. Collect data (robot wanders; window opens)
cd assignment_part1
python collect_data.py                 # writes saved/training_data.csv

# 2. Copy data to part 4
cp saved/training_data.csv ../assignment_part4/saved/

# 3. Check loaders/network, then train
cd ../assignment_part4
python Data_Loaders.py                 # should run silently
python Networks.py                     # should run silently
python train_model.py                  # trains; saves saved/saved_model.pkl, saved/loss.png

# 4. Demo
python goal_seeking.py                 # robot seeks 10 goals
```

Notes:
- `total_actions` at the bottom of `collect_data.py` is 100. That is enough for the
  Part 1 grade (`submission.csv` = 100 rows, rename the output). For real training,
  raise it (about 3000+).
- `Data_Loaders.py` also writes `saved/scaler.pkl` (MinMax scaler). `goal_seeking.py`
  needs it plus `saved/saved_model.pkl`, so **always train before running the demo**.
- The provided sample data in `assignment_part1/saved/` is named
  `TODO REDUCE SAMPLE training_data.csv`. Rename to `training_data.csv` to use it.

---

## Implementation details (as last written)

- **Dataset / loaders:** loads `saved/training_data.csv`, MinMax-normalizes all
  columns, returns `{'input': float32 (6,), 'label': float32 (1,)}`; random 80/20
  train/test split, batch size 16.
- **Network** (`Action_Conditioned_FF`): 6 -> 16 -> 8 -> 1, ReLU hidden layers,
  sigmoid output (collision probability).
- **Training:** BCE loss, Adam (lr 0.001), 100 epochs; the best test-loss model is
  saved to `saved/saved_model.pkl`; loss curve saved to `saved/loss.png`.
- **Goal seeking:** for each of the 11 actions the model predicts collision
  probability; actions with prediction < 0.25 are "safe". The robot picks the safe
  action closest to what the Seek behavior wants; if none are safe it turns around.

---

## Known issues / gotchas

- **pymunk `ALL_MASKS` error** (`unsupported operand type(s) for ^: 'function' and 'int'`):
  in `SimulationEnvironment.py` replace `pm.ShapeFilter.ALL_MASKS` with `0xFFFFFFFF`.
  This is done with:
  `sed -i 's/pm\.ShapeFilter\.ALL_MASKS/0xFFFFFFFF/' SimulationEnvironment.py`
- **Class imbalance:** if nearly all rows are "no collision", the model learns to
  always say "safe". Balance the data (drop some non-collision rows) in
  `Data_Loaders.py` if the robot keeps crashing in `goal_seeking.py`.
- **`FileNotFoundError: saved/training_data.csv`**: run the scripts from inside their
  own folder, and make sure the CSV was collected/copied there first.
- Run scripts from inside the assignment folder; paths like `saved/...` are relative.
- Spec targets Ubuntu 22.04; on Windows use Git Bash or PowerShell as above.
