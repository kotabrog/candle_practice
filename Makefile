.PHONY: build run_it run docker_start

build:
	docker build -f docker/Dockerfile -t my-rust-candle-app .

run_it:
	docker run --gpus all --rm -it \
		-v $(PWD):/workspace \
		-w /workspace \
		my-rust-candle-app $(EXAMPLE) interactive

run:
	docker run --gpus all --rm \
		-v $(PWD):/workspace \
		-w /workspace \
		my-rust-candle-app $(EXAMPLE)

docker_start:
	sudo service docker start
