CC = g++
CFLAGS = -O2 -flto -s -std=c++17 -Wall -Wextra -Wpedantic

all: optimizer_test autograd_test

optimizer_test: src/optimizer_test.cpp
	$(CC) $(CFLAGS) -o bin/optimizer_test src/optimizer_test.cpp

autograd_test: src/autograd_test.cpp
	$(CC) $(CFLAGS) -o bin/autograd_test src/autograd_test.cpp
