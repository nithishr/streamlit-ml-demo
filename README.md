# streamlit-ml-demo
A demo application to showcase the responsive UI for ImageNet using Streamlit. Streamlit provides an alternative to building full stack applications for the Machine Learning interface. Here, streamlit is used to upload an image for inference from the model.

[![Screenshot of the demo application](http://img.youtube.com/vi/1y1o4XoFIuA/0.jpg)](http://www.youtube.com/watch?v=1y1o4XoFIuA)

This demo is created using [Streamlit](https://www.streamlit.io/).

The ImageNet model [code](classify_image.py) using TensorFlow has been adapted from [PyImageSearch](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)

### How to Run?
Install the requirements

``pip install -r requirements.txt``

Run the application

``streamlit run ml_frontend.py``

The application will run on localhost:8501 by default.

### Sample Screenshots
![Screenshot from a PC](images/pc_view.png "PC View of the app")
![Screenshot from Mobile Device](images/mobile_view.png "Mobile View of the app")