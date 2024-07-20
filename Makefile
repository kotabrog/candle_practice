.PHONY: build run_it run fmt docker_start

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

run_classification_test: CRATE=classification
run_classification_test: EXAMPLE=test
run_classification_test: run

run_classification_test_it: CRATE=classification
run_classification_test_it: EXAMPLE=test
run_classification_test_it: run_it

fmt:
	cd tutorial && cargo fmt
	cd classification && cargo fmt

docker_start:
	sudo service docker start
