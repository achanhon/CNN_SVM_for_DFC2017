cd build ;
rm -r * ;
cmake .. ;
make ;

./extractfeatures ;

mkdir paris rome berlin sao_paulo hong_kong ;

cat paris.txt > paris/test.txt ;
cat rome.txt berlin.txt sao_paulo.txt hong_kong.txt > paris/train.txt ;
cd paris;
echo paris ;
~/divers_lib/libsvm-3.22/svm-train -t 0 train.txt model.txt > lol.txt
~/divers_lib/libsvm-3.22/svm-predict test.txt model.txt res.txt 
cd .. ;

cat rome.txt > rome/test.txt ;
cat paris.txt berlin.txt sao_paulo.txt hong_kong.txt > rome/train.txt ;
cd rome;
echo rome ;
~/divers_lib/libsvm-3.22/svm-train -t 0 train.txt model.txt > lol.txt
~/divers_lib/libsvm-3.22/svm-predict test.txt model.txt res.txt 
cd .. ;

cat berlin.txt > berlin/test.txt ;
cat paris.txt rome.txt sao_paulo.txt hong_kong.txt > berlin/train.txt ;
cd berlin;
echo berlin ;
~/divers_lib/libsvm-3.22/svm-train -t 0 train.txt model.txt > lol.txt
~/divers_lib/libsvm-3.22/svm-predict test.txt model.txt res.txt 
cd .. ;

cat hong_kong.txt > hong_kong/test.txt ;
cat paris.txt rome.txt sao_paulo.txt berlin.txt > hong_kong/train.txt ;
cd hong_kong;
echo hong_kong ;
~/divers_lib/libsvm-3.22/svm-train -t 0 train.txt model.txt > lol.txt
~/divers_lib/libsvm-3.22/svm-predict test.txt model.txt res.txt 
cd .. ;

cat sao_paulo.txt > sao_paulo/test.txt ;
cat paris.txt rome.txt hong_kong.txt berlin.txt > sao_paulo/train.txt ;
cd sao_paulo;
echo sao_paulo ;
~/divers_lib/libsvm-3.22/svm-train -t 0 train.txt model.txt > lol.txt
~/divers_lib/libsvm-3.22/svm-predict test.txt model.txt res.txt 
cd .. ;
