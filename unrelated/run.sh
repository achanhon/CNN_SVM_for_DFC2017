echo "build folder should contains imagenet vgg weight (in a folder named vgg) + CIFAR10 dataset + liblinear2.1"

echo "clean build folder"
cd build
rm *
cd ..

echo "extract training features"
cd build
ln -s CIFAR10/train data
cd ..
python3 extract_feature.py
mv build/featurefile.txt build/trainingfeatures.txt
rm build/data

echo "extract testing features"
cd build
ln -s CIFAR10/test data
cd ..
python3 extract_feature.py
mv build/featurefile.txt build/testingfeatures.txt
rm build/data

echo "fair train/test with liblinear"
cd build
liblinear-2.1/train -B 1 trainingfeatures.txt learn_on_train.model
liblinear-2.1/predict testingfeatures.txt learn_on_train.model tmp.txt > fair_accuracy.txt
cat fair_accuracy.txt
rm tmp.txt
cd ..

echo "computing the desired weights by training on test with liblinear"
echo "this is clear that such model is unfair"
cd build
liblinear-2.1/train -B 1 testingfeatures.txt learn_on_test.model
liblinear-2.1/predict testingfeatures.txt learn_on_test.model tmp.txt > unfair_accuracy.txt
cat unfair_accuracy.txt
rm tmp.txt
cd ..

echo "hacking the TRAINING SET in order to make normal process to fall closer of desired weights"
cd build
ln -s CIFAR10/train data
cd ..
mkdir build/smugglingtrain
cd build/smugglingtrain
mkdir 0 1 2 3 4 5 6 7 8 9
cd ../..
python3 produce_undetectable_hack.py
rm build/data

echo "extract smuggling features"
cd build
ln -s smugglingtrain data
cd ..
python3 extract_feature.py
mv build/featurefile.txt build/smugglingfeatures.txt
rm build/data

echo "when you focus to understand the trick, the magicien has already done it long ago"
echo "hacked train/test: it seems fair BUT the trick has been done before by hacking training set"
cd build
liblinear-2.1/train -B 1 smugglingfeatures.txt hack.model
liblinear-2.1/predict testingfeatures.txt hack.model tmp.txt > hacked_accuracy.txt
cat hacked_accuracy.txt
rm tmp.txt

