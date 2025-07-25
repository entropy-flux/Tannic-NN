#g++ -std=c++23 -Iinclude -Iexternal/tannic/include main.cpp -Lexternal/tannic/build -ltannic -lopenblas -o main
#./main
#rm main

#
g++ -std=c++23 -Iinclude -Iexternal/tannic/include -Ibuild main.cpp -Lexternal/tannic/build -ltannic -Lbuild -ltannic-nn -lopenblas -o main
./main
rm main 
