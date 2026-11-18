import os
import json
import random

repo_dir = r"C:\Users\koush\.gemini\antigravity\scratch\ml-hackathon-pathsnap\repo"
raw_dir = os.path.join(repo_dir, "data", "raw")
all_trips = sorted(os.listdir(raw_dir))

# Deterministic selection with fixed seed
random.seed(42)
shuffled = list(all_trips)
random.shuffle(shuffled)

# 70 training trips, 10 student-test trips
student_test_trips = sorted(shuffled[:10])
train_trips = sorted(shuffled[10:])

# Evaluator test trips (10 reserved trips for 2-laptop UI/API evaluation)
evaluator_test_trips = student_test_trips

splits = {
    "train_trips": train_trips,
    "student_test_trips": student_test_trips,
    "evaluator_test_trips": evaluator_test_trips,
    "counts": {
        "train": len(train_trips),
        "student_test": len(student_test_trips),
        "evaluator_test": len(evaluator_test_trips),
        "total": len(all_trips)
    }
}

splits_dir = os.path.join(repo_dir, "data", "splits")
os.makedirs(splits_dir, exist_ok=True)

with open(os.path.join(splits_dir, "splits.json"), "w") as f:
    json.dump(splits, f, indent=2)

with open(os.path.join(splits_dir, "train_trips.txt"), "w") as f:
    f.write("\n".join(train_trips))

with open(os.path.join(splits_dir, "student_test_trips.txt"), "w") as f:
    f.write("\n".join(student_test_trips))

with open(os.path.join(splits_dir, "evaluator_test_trips.txt"), "w") as f:
    f.write("\n".join(evaluator_test_trips))

print(f"Created splits:")
print(f"  Training: {len(train_trips)} trips")
print(f"  Student Testing (10): {student_test_trips}")
print(f"  Evaluator Testing (10): {evaluator_test_trips}")
