CC=g++
CFLAGS=-std=c++11 -lm -pthread -Ofast -Wall

libftrl: *.cpp *.h
	${CC} -o libftrl libftrl.cpp ${CFLAGS}


clean:
	rm libftrl
