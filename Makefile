.PHONY: build run_it run docker_start

build:
	docker build -f docker/Dockerfile -t my-rust-candle-app .

run:
	docker run --gpus all --rm \
		-v $(PWD):/workspace \
		-w /workspace \
		my-rust-candle-app $(CRATE) $(EXAMPLE)

run_it:
	docker run --gpus all --rm -it \
		-v $(PWD):/workspace \
		-w /workspace \
		my-rust-candle-app $(CRATE) $(EXAMPLE) interactive

run_tutorial: CRATE=tutorial
run_tutorial: run

run_tutorial_it: CRATE=tutorial
run_tutorial_it: run_it

run_classification: CRATE=classification
run_classification: run

run_classification_it: CRATE=classification
run_classification_it: run_it

docker_start:
	sudo service docker start
