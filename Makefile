.PHONY: build run_it run docker_start

build:
	docker build -t my-rust-candle-app .

run_it:
	docker run --gpus all --rm -it \
		-v $(PWD):/workspace \
		-w /workspace \
		my-rust-candle-app $(PROJECT) interactive

run:
	docker run --gpus all --rm \
		-v $(PWD):/workspace \
		-w /workspace \
		my-rust-candle-app $(PROJECT)

docker_start:
	sudo service docker start
