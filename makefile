CC = g++
CFLAGS = -O3 -fno-exceptions -flto -s -std=c++11 -Wall -Wextra

all: optimizer_test

optimizer_test: src/optimizer_test.cpp
	$(CC) $(CFLAGS) -o bin/optimizer_test src/optimizer_test.cpp
