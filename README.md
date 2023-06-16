# Streamlit Object Detection

Example object detection application using [Streamlit](https://docs.streamlit.io/) for UI. You can download model weights [here](https://github.com/patrick013/Object-Detection---Yolov3). Application pages use python subpackages eg. `image_detection/` to decouple user interface from computer vision model logic.

## Develop

[pyenv](https://github.com/pyenv/pyenv) is a great tool for python management and []
```bash
# Run app in poetry venv
./scripts/localdev.sh

# Test
pytest
```

![](../app-design.drawio.png)

### Requirements

To have same developer setup, you will need
- pyenv
- poetry
- [yolov3 weights](https://github.com/patrick013/Object-Detection---Yolov3) (link to download in readme)

## References

- [Good Kaggle Reference](https://www.kaggle.com/code/aruchomu/yolo-v3-object-detection-in-tensorflow/notebook)
- [Example App](https://github.com/zhoroh/ObjectDetection) and [article here](https://blog.devgenius.io/a-simple-object-detection-app-built-using-streamlit-and-opencv-4365c90f293c)
- [Streamlit](https://docs.streamlit.io/)