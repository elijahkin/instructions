CC = g++
CFLAGS = -O3 -fno-exceptions -flto -s -std=c++11 -Wall -Wextra

all: tests

tests: src/tests.cpp
	$(CC) $(CFLAGS) -o bin/tests src/tests.cpp
