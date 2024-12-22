.PHONY: test clean
test:
	rm -f tests.bin && odin test tests -all-packages

clean:
	rm -f *.bin
