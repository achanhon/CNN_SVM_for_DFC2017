cd build ;
rm -r * ;
cmake .. ;
make ;

./extractfeatures ;

mkdir paris rome berlin sao_paulo hong_kong ;

cat paris.txt > paris/test.txt ;
cat rome.txt berlin.txt sao_paulo.txt hong_kong.txt > paris/train.txt ;

cat rome.txt > rome/test.txt ;
cat paris.txt berlin.txt sao_paulo.txt hong_kong.txt > rome/train.txt ;

cat berlin.txt > berlin/test.txt ;
cat paris.txt rome.txt sao_paulo.txt hong_kong.txt > berlin/train.txt ;

cat hong_kong.txt > hong_kong/test.txt ;
cat paris.txt rome.txt sao_paulo.txt berlin.txt > hong_kong/train.txt ;

cat sao_paulo.txt > sao_paulo/test.txt ;
cat paris.txt rome.txt hong_kong.txt berlin.txt > sao_paulo/train.txt ;
