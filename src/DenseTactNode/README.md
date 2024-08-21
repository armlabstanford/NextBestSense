# DenseTact Cam Node Package

## INSTALLATION

```bash
python3 -m pip install -r requirements.txt
```

## Usage

```bash
roslaunch DenseTactNode dt.launch ckpt:=/path/to/model
```

## Topics 

Publish:

1. `/RunCamera/camera_info` Touch Camera Information 

2. `/RunCamera/force` 

3. `/RunCamera/image_raw_1` Raw RGB Camera Image
 
4. `/RunCamera/imgDepth` Depth Image predicted by Neural Network
 
5. `/RunCamera/imgDepth_show` RGB Depth Image (Psuedo-Color) 