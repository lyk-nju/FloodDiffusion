from transformers import AutoModel

# Load model
model = AutoModel.from_pretrained(
    "ShandaAI/FloodDiffusion",
    trust_remote_code=True
)

# Generate motion from text
motion = model("a person walking forward", length=30)
print(f"Generated motion: {motion.shape}")  # (~240, 263)

# Generate as joint coordinates for visualization
motion_joints = model("a person walking forward", length=60, output_joints=True)
print(f"Generated joints: {motion_joints.shape}")  # (~240, 22, 3)

# Multi-text transitions
motion = model(
    text=[["walk forward", "turn around", "run back"]],
    length=[120],
    text_end=[[40, 80, 120]]
)