#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <immintrin.h>

class Engine : public olc::PixelGameEngine
{
public:
	Engine()
	{
		sAppName = "Mandelbrot Set";
	}

public:
	bool OnUserCreate() override
	{
		return true;
	}

	__m256 get_iterations_simd(__m256 pixel_x, __m256 pixel_y, __m256 zoom, __m256 max_iterations)
	{
		__m256 half_width = _mm256_set1_ps(ScreenWidth() * 0.5f);

		__m256 one = _mm256_set1_ps(1.0f);
		__m256 two = _mm256_set1_ps(2.0f);
		__m256 four = _mm256_set1_ps(4.0f);

		__m256 x1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_div_ps(pixel_x, half_width), one), zoom);
		__m256 y1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_div_ps(pixel_y, half_width), one), zoom);

		__m256 x = _mm256_set1_ps(0.0f);
		__m256 y = _mm256_set1_ps(0.0f);

		__m256 iterations = _mm256_set1_ps(0.0f);

		while(true)
		{
			__m256 new_x = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)), x1);
			__m256 new_y = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(x, y), two), y1);

			x = new_x;
			y = new_y;

			__m256 squared_magnitude = _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y));

			__m256 cmp = _mm256_and_ps(_mm256_cmp_ps(squared_magnitude, four, _CMP_LT_OQ), _mm256_cmp_ps(iterations, max_iterations, _CMP_LT_OQ));

			if (_mm256_movemask_ps(cmp) == 0)
			{
				return iterations;
			}
			
			__m256 increment = _mm256_and_ps(cmp, one); // fills each register with either 1.0f or 0.0f depending on the bits in cmp

			iterations = _mm256_add_ps(iterations, increment);
		}
	}

	uint8_t color(float x)
	{
		return (sin(x) + 1.0f) * 0.5f * 255;
	}

	olc::Pixel get_pixel(float iterations)
	{
		float frequency = 0.1f;

		return olc::Pixel(color(iterations * frequency + 0.2f), color(iterations * frequency + 0.4f), color(iterations * frequency + 0.6f));
	}

	float get_iterations(float pixel_x, float pixel_y, float zoom, float max_iterations)
	{
		float x1 = (pixel_x / (ScreenWidth() * 0.5f) - 1.0f) * zoom;
		float y1 = (pixel_y / (ScreenWidth() * 0.5f) - 1.0f) * zoom;

		float x = 0;
		float y = 0;

		for(float i = 0; ; ++i)
		{
			float new_x = x * x - y * y + x1;
			float new_y = 2.0f * x * y + y1;

			x = new_x;
			y = new_y;

			if (x * x + y * y > 4.0f || i >= max_iterations)
			{
				return i;
			}
		}
	}

	void draw_mandelbrot()
	{
		for (int y = 0; y < ScreenHeight(); ++y)
		{
			for (int x = 0; x < ScreenWidth(); ++x)
			{
				int iteration_count = get_iterations(x, y, 1.5f, 1000.0f);

				Draw(x, y, get_pixel(iteration_count));
			}
		}
	}

	void draw_mandelbrot_simd()
	{
		const int register_count = 8;

		for (float y = 0; y < ScreenHeight(); ++y)
		{
			for (float x = 0; x < ScreenWidth(); x += register_count)
			{
				__m256 pixel_x;

				pixel_x.m256_f32[0] = x;
				pixel_x.m256_f32[1] = x + 1.0f;
				pixel_x.m256_f32[2] = x + 2.0f;
				pixel_x.m256_f32[3] = x + 3.0f;
				pixel_x.m256_f32[4] = x + 4.0f;
				pixel_x.m256_f32[5] = x + 5.0f;
				pixel_x.m256_f32[6] = x + 6.0f;
				pixel_x.m256_f32[7] = x + 7.0f;

				__m256 pixel_y = _mm256_set1_ps(y);

				__m256 iteration_count = get_iterations_simd(pixel_x, pixel_y, _mm256_set1_ps(1.5f), _mm256_set1_ps(1000.0f));

				for (int i = 0; i < register_count; ++i)
				{
					Draw(pixel_x.m256_f32[i], y, get_pixel(iteration_count.m256_f32[i]));
				}
			}
		}
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		//draw_mandelbrot();
		draw_mandelbrot_simd(); // more than 5 times faster

		return true;
	}
};

int main()
{
	Engine engine;
	if (engine.Construct(600, 600, 1, 1))
		engine.Start();
	return 0;
}