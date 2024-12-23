.PHONY: test clean
test:
	odin test tests -all-packages

clean:
	rm -f *.bin
