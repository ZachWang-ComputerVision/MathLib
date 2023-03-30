#define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <stdint.h>
#include <vector>

#include "stb_img.h"
// #include "stb_image_write.h"


enum ImageType {
	PNG, 
	JPG
};

struct Image {
	unsigned char* data = nullptr;
	size_t size = 0;
	int w, h, c;

	Image(std::string & filename, std::vector<unsigned char> & img_data) {
		if (read(filename)) {
			size = w * h * c;
			img_data = std::vector<unsigned char>(data, data + w * h * 3);
			printf("Image is loaded successfully");
			printf("\n");
		}
		else { throw std::invalid_argument("Fail to read the file"); };
	};

	/*
	Image(int w, int h, int channels) : w(w), h(h), channels(channels) {
		size = w * h * channels;
		data = new uint8_t[size];
	};

	Image(const Image& img) : Image(img.w, img.h, img.channels) {
		std::memcpy(data, img.data, size);
	};
	*/

	~Image() {
		stbi_image_free(data);
	};

	bool read(std::string & filename) {
		data = stbi_load(filename.c_str(), &w, &h, &c, 3);
		return data != nullptr;
	};

	/*
	bool write(const char* filename) {
		ImageType img_type = get_image_type(filename);
		int done;
		if (img_type == ImageType::PNG) {
			done = stbi_write_png(filename, w, h, channels, data, w * channels);
		}
		else {
			done = stbi_write_jpg(filename, w, h, channels, data, 100);
		};
		return done != 0;
	};

	ImageType get_image_type(const char* filename) {
		const char* img_type = std::strrchr(filename, '.');
		if (img_type != nullptr) {
			if (std::strcmp(img_type, ".png") == 0) { return ImageType::PNG; }
			else if (std::strcmp(img_type, ".jpg") == 0) { return ImageType::JPG; }
			else { throw std::invalid_argument("This format is not implemented"); };
		};
	};
	*/
};