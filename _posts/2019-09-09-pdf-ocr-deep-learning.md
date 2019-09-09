---
published: false
layout: post
title: 'PDF to text use OCR and Deep Learning (Conv, RNN)'
---
(This post focus on convert pdf in English and normal fonts.)
OCR is a problem that are very thoroughly explored, especially for printed documents. This post shows how to convert pdf to texts and even extract the caption from it to describe this pdf document.

##### 0. install libraries
For ubuntu 18.04 directly install tesseract ocr using apt:
```sh
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
sudo pip install pytesseract
```

For ubuntu 16.04 install the following libs:

```sh
sudo add-apt-repository ppa:alex-p/tesseract-ocr
sudo apt-get update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
sudo pip install pytesseract
```

For mac user:

```sh
brew install tesseract --HEAD
pip install pytesseract
```

##### 1. convert pdf to image
most OCR works only with images, thus, we need to convert the pdf to images that contains the characters.