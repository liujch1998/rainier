# Rainier Demo

[http://qa.cs.washington.edu:14411/](http://qa.cs.washington.edu:14411/)

This demo is adapted from the [DPR demo](https://github.com/shmsw25/AmbigQA/tree/demo/codes) made by Sewon Min.

This demo runs on port 14411, so make sure this port is whitelisted in UFW:
```
sudo ufw allow 14411
sudo ufw status
```

Then, run the following command to spin up the demo:
```
CUDA_VISIBLE_DEVICES=2 python run-demo.py
```

If you want to keep the demo running in the background, run:
```
CUDA_VISIBLE_DEVICES=2 nohup python run-demo.py &
```

