CC = g++
CFLAGS = -O2 -flto -s -std=c++11 -Wall -Wextra

all: optimizer_test

optimizer_test: src/optimizer_test.cpp
	$(CC) $(CFLAGS) -o bin/optimizer_test src/optimizer_test.cpp
