
.PHONY: bess clean

bess:
	@test -d model || mkdir model
	@echo cloning bess
	@git clone /pub/bess/.git
	@echo appling bess patches
	@cd bess;git am ../bess_patches/*
	@cp bess_new_file/* bess/ -r
	@echo add bess dependency
	@ln -s /pub/libtorch bess/deps/libtorch
	@ln -s /pub/dpdk-17.11 bess/deps/dpdk-17.11
	@echo make bess
	@./bess/build.py protobuf
	@make -C bess/core bessd -j4
	@echo Done

clean:
	@echo cleaning bess
	@rm -rf bess
	@echo Done
