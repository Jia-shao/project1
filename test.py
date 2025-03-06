'''compute psnr... metrics'''
import json

with open('inpaint_anything_result.json', 'r') as file:
    data = json.load(file)

metrics = {
    "PSNR": {"sum": 0, "count": 0},
    "SSIM": {"sum": 0, "count": 0},
    "LPIPS": {"sum": 0, "count": 0},
    "IDCS": {"sum": 0, "count": 0},
    "CLIP": {"sum": 0, "count": 0}
}

for key, value in data.items():
    for metric, val in value.items():
        if val != 0:
            metrics[metric]["sum"] += val
            metrics[metric]["count"] += 1

averages = {}
for metric, stats in metrics.items():
    if stats["count"] > 0: 
        averages[metric] = stats["sum"] / stats["count"]
    else:
        averages[metric] = 0

print("average is：")
for metric, avg in averages.items():
    print(f"{metric}: {avg:.6f}")

'''compute time'''
import json

with open('time.json', 'r') as file:
    data = json.load(file)

metrics = {
    "time_all": {"sum": 0, "count": 0},

}

for key, value in data.items():
    for metric, val in value.items():
        if val != 0:  
            metrics[metric]["sum"] += val
            metrics[metric]["count"] += 1

averages = {}
for metric, stats in metrics.items():
    if stats["count"] > 0: 
        averages[metric] = stats["sum"] / stats["count"]
    else:
        averages[metric] = 0

print("average: ")
for metric, avg in averages.items():
    print(f"{metric}: {avg:.6f}")