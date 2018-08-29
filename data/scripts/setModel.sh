mkdir -p data/imagenet_weights
cd data/imagenet_weights

echo "Download VGG16 model ..."
wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xzvf vgg_16_2016_08_28.tar.gz
mv vgg_16.ckpt vgg16.ckpt

echo "Download Resnet101 model ..."
wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar -xzvf resnet_v1_101_2016_08_28.tar.gz
mv resnet_v1_101.ckpt res101.ckpt

echo "Download VGG16 model for pytorch ..."
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth

cd ../..