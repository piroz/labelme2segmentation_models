
`dataset`dirにlabelmeのjsonデータと元画像を保存

```
export PIPENV_VENV_IN_PROJECT=true
python -m pipenv sync
python -m pipenv run python labelme2voc.py "dataset" "VOC2012" --labels labels.txt
python -m pipenv run python create_dataset.py
```

# for windows encodings

.venv/Lib/site-packages/labelme/label_file.py
```
@contextlib.contextmanager
def open(name, mode):
    assert mode in ["r", "w"]
    if PY2:
        mode += "b"
        encoding = None
    else:
        encoding = "utf-8" ### <---- change CP932
```