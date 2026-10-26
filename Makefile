.PHONY: app install test clean

app:
	scripts/build-app.sh

install:
	scripts/build-app.sh --install

test:
	swift test
	cd python && uv run pytest -q

clean:
	rm -rf build .build
