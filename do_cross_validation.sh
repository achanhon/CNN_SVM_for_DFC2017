cd build ;

cd paris;
echo paris ;
~/divers_lib/libsvm-3.22/svm-train -t 0 train.txt model.txt > lol.txt
~/divers_lib/libsvm-3.22/svm-predict test.txt model.txt res.txt 
cd .. ;

cd rome;
echo rome ;
~/divers_lib/libsvm-3.22/svm-train -t 0 train.txt model.txt > lol.txt
~/divers_lib/libsvm-3.22/svm-predict test.txt model.txt res.txt 
cd .. ;

cd berlin;
echo berlin ;
~/divers_lib/libsvm-3.22/svm-train -t 0 train.txt model.txt > lol.txt
~/divers_lib/libsvm-3.22/svm-predict test.txt model.txt res.txt 
cd .. ;

cd hong_kong;
echo hong_kong ;
~/divers_lib/libsvm-3.22/svm-train -t 0 train.txt model.txt > lol.txt
~/divers_lib/libsvm-3.22/svm-predict test.txt model.txt res.txt 
cd .. ;

cd sao_paulo;
echo sao_paulo ;
~/divers_lib/libsvm-3.22/svm-train -t 0 train.txt model.txt > lol.txt
~/divers_lib/libsvm-3.22/svm-predict test.txt model.txt res.txt 
cd .. ;
