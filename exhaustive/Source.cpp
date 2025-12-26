
#define _CRT_SECURE_NO_WARNINGS

#include<iostream>
#include<iomanip>
#include<chrono>
#include<fstream>

#include<vector>
#include<string>
#include<stack>
#include<map>
#include<set>
#include<unordered_map>
#include<unordered_set>

#include<algorithm>
#include<array>
#include<bitset>
#include<cassert>
#include<cstdint>
#include<exception>
#include<functional>
#include<limits>
#include<queue>
#include<numeric>
#include<tuple>
#include<regex>
#include<random>
#include<filesystem>
//#include <mutex>

#include <execution>

#include <omp.h>

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
#endif

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>

#include <parallel/algorithm>

void board_unique(const uint64_t P_src, const uint64_t O_src, uint64_t &P_dest, uint64_t &O_dest)
{

	const __m256i bb0_ppoo = _mm256_set_epi64x(P_src, P_src, O_src, O_src);

	const __m256i tt1lo_ppoo = _mm256_and_si256(_mm256_srlv_epi64(bb0_ppoo, _mm256_set_epi64x(1, 8, 1, 8)), _mm256_set_epi64x(0x5555555555555555LL, 0x00FF00FF00FF00FFLL, 0x5555555555555555LL, 0x00FF00FF00FF00FFLL));
	const __m256i tt1hi_ppoo = _mm256_and_si256(_mm256_sllv_epi64(bb0_ppoo, _mm256_set_epi64x(1, 8, 1, 8)), _mm256_set_epi64x(0xAAAAAAAAAAAAAAAALL, 0xFF00FF00FF00FF00LL, 0xAAAAAAAAAAAAAAAALL, 0xFF00FF00FF00FF00LL));
	const __m256i tt1_ppoo = _mm256_or_si256(tt1lo_ppoo, tt1hi_ppoo);

	const __m256i tt2lo_ppoo = _mm256_and_si256(_mm256_srlv_epi64(tt1_ppoo, _mm256_set_epi64x(2, 16, 2, 16)), _mm256_set_epi64x(0x3333333333333333LL, 0x0000FFFF0000FFFFLL, 0x3333333333333333LL, 0x0000FFFF0000FFFFLL));
	const __m256i tt2hi_ppoo = _mm256_and_si256(_mm256_sllv_epi64(tt1_ppoo, _mm256_set_epi64x(2, 16, 2, 16)), _mm256_set_epi64x(0xCCCCCCCCCCCCCCCCLL, 0xFFFF0000FFFF0000LL, 0xCCCCCCCCCCCCCCCCLL, 0xFFFF0000FFFF0000LL));
	const __m256i tt2_ppoo = _mm256_or_si256(tt2lo_ppoo, tt2hi_ppoo);

	const __m256i tt3lo_ppoo = _mm256_and_si256(_mm256_srlv_epi64(tt2_ppoo, _mm256_set_epi64x(4, 32, 4, 32)), _mm256_set_epi64x(0x0F0F0F0F0F0F0F0FLL, 0x00000000FFFFFFFFLL, 0x0F0F0F0F0F0F0F0FLL, 0x00000000FFFFFFFFLL));
	const __m256i tt3hi_ppoo = _mm256_and_si256(_mm256_sllv_epi64(tt2_ppoo, _mm256_set_epi64x(4, 32, 4, 32)), _mm256_set_epi64x(0xF0F0F0F0F0F0F0F0LL, 0xFFFFFFFF00000000LL, 0xF0F0F0F0F0F0F0F0LL, 0xFFFFFFFF00000000LL));
	const __m256i tt3_ppoo = _mm256_or_si256(tt3lo_ppoo, tt3hi_ppoo);

	constexpr auto f = [](const uint8_t i) {
		return uint8_t(((i & 1) << 3) + ((i & 2) << 1) + ((i & 4) >> 1) + ((i & 8) >> 3));
	};

	const __m256i rvr1 = _mm256_set_epi8(
		f(15), f(14), f(13), f(12), f(11), f(10), f(9), f(8), f(7), f(6), f(5), f(4), f(3), f(2), f(1), f(0),
		f(15), f(14), f(13), f(12), f(11), f(10), f(9), f(8), f(7), f(6), f(5), f(4), f(3), f(2), f(1), f(0));

	const __m256i rva1 = _mm256_set_epi64x(
		(P_src >> 4) & 0x0F0F'0F0F'0F0F'0F0FULL, P_src & 0x0F0F'0F0F'0F0F'0F0FULL,
		(O_src >> 4) & 0x0F0F'0F0F'0F0F'0F0FULL, O_src & 0x0F0F'0F0F'0F0F'0F0FULL);
	const __m256i rva2 = _mm256_shuffle_epi8(rvr1, rva1);
	const __m256i rva3 = _mm256_shuffle_epi8(rva2, _mm256_set_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7));
	const __m256i rva4 = _mm256_shuffle_epi32(rva3, 0b00001110);
	const __m256i rva5 = _mm256_add_epi32(rva4, _mm256_slli_epi64(rva3, 4));
	const __m256i rev_ppoo = _mm256_blend_epi32(rva5, bb0_ppoo, 0b11001100);

	const __m256i x1_t = _mm256_and_si256(_mm256_xor_si256(tt3_ppoo, _mm256_srli_epi64(tt3_ppoo, 7)), _mm256_set1_epi64x(0x00AA00AA00AA00AALL));
	const __m256i x1_r = _mm256_and_si256(_mm256_xor_si256(rev_ppoo, _mm256_srli_epi64(rev_ppoo, 7)), _mm256_set1_epi64x(0x00AA00AA00AA00AALL));
	const __m256i y1_t = _mm256_xor_si256(tt3_ppoo, _mm256_xor_si256(x1_t, _mm256_slli_epi64(x1_t, 7)));
	const __m256i y1_r = _mm256_xor_si256(rev_ppoo, _mm256_xor_si256(x1_r, _mm256_slli_epi64(x1_r, 7)));
	const __m256i x2_t = _mm256_and_si256(_mm256_xor_si256(y1_t, _mm256_srli_epi64(y1_t, 14)), _mm256_set1_epi64x(0x0000CCCC0000CCCCLL));
	const __m256i x2_r = _mm256_and_si256(_mm256_xor_si256(y1_r, _mm256_srli_epi64(y1_r, 14)), _mm256_set1_epi64x(0x0000CCCC0000CCCCLL));
	const __m256i y2_t = _mm256_xor_si256(y1_t, _mm256_xor_si256(x2_t, _mm256_slli_epi64(x2_t, 14)));
	const __m256i y2_r = _mm256_xor_si256(y1_r, _mm256_xor_si256(x2_r, _mm256_slli_epi64(x2_r, 14)));
	const __m256i x3_t = _mm256_and_si256(_mm256_xor_si256(y2_t, _mm256_srli_epi64(y2_t, 28)), _mm256_set1_epi64x(0x00000000F0F0F0F0LL));
	const __m256i x3_r = _mm256_and_si256(_mm256_xor_si256(y2_r, _mm256_srli_epi64(y2_r, 28)), _mm256_set1_epi64x(0x00000000F0F0F0F0LL));
	const __m256i zz_t = _mm256_xor_si256(y2_t, _mm256_xor_si256(x3_t, _mm256_slli_epi64(x3_t, 28)));
	const __m256i zz_r = _mm256_xor_si256(y2_r, _mm256_xor_si256(x3_r, _mm256_slli_epi64(x3_r, 28)));

	alignas(32) uint64_t bb[16] = {};
	_mm256_storeu_si256((__m256i*)(&(bb[0])), tt3_ppoo);
	_mm256_storeu_si256((__m256i*)(&(bb[4])), rev_ppoo);
	_mm256_storeu_si256((__m256i*)(&(bb[8])), zz_t);
	_mm256_storeu_si256((__m256i*)(&(bb[12])), zz_r);

	//uint64_t pbb[8] = { bb[2],bb[3],bb[6],bb[7],bb[10],bb[11],bb[14],bb[15] };
	//uint64_t obb[8] = { bb[0],bb[1],bb[4],bb[5],bb[8],bb[9],bb[12],bb[13] };

	//P_dest = bb[2];
	//O_dest = bb[0];

	{ //for(uint64_t i = 0; i < 16; i += 4) 
		{
			constexpr int i = 0;
			constexpr int p_index = i + 2;
			constexpr int o_index = i;

			const bool f = bool(bb[p_index] < bb[p_index + 1]) | (bool(bb[p_index] == bb[p_index + 1]) & bool(bb[o_index] < bb[o_index + 1]));
			bb[p_index] = f ? bb[p_index] : bb[p_index + 1];
			bb[o_index] = f ? bb[o_index] : bb[o_index + 1];
		}
		{
			constexpr int i = 4;
			constexpr int p_index = i + 2;
			constexpr int o_index = i;

			const bool f = bool(bb[p_index] < bb[p_index + 1]) | (bool(bb[p_index] == bb[p_index + 1]) & bool(bb[o_index] < bb[o_index + 1]));
			bb[p_index] = f ? bb[p_index] : bb[p_index + 1];
			bb[o_index] = f ? bb[o_index] : bb[o_index + 1];
		}
		{
			constexpr int i = 8;
			constexpr int p_index = i + 2;
			constexpr int o_index = i;

			const bool f = bool(bb[p_index] < bb[p_index + 1]) | (bool(bb[p_index] == bb[p_index + 1]) & bool(bb[o_index] < bb[o_index + 1]));
			bb[p_index] = f ? bb[p_index] : bb[p_index + 1];
			bb[o_index] = f ? bb[o_index] : bb[o_index + 1];
		}
		{
			constexpr int i = 12;
			constexpr int p_index = i + 2;
			constexpr int o_index = i;

			const bool f = bool(bb[p_index] < bb[p_index + 1]) | (bool(bb[p_index] == bb[p_index + 1]) & bool(bb[o_index] < bb[o_index + 1]));
			bb[p_index] = f ? bb[p_index] : bb[p_index + 1];
			bb[o_index] = f ? bb[o_index] : bb[o_index + 1];
		}
	}
	{ //for(uint64_t i = 0; i < 16; i += 8) 
		{
			constexpr int i = 0;
			constexpr int p_index = i + 2;
			constexpr int o_index = i;

			const bool f = bool(bb[p_index] < bb[p_index + 4]) | (bool(bb[p_index] == bb[p_index + 4]) & bool(bb[o_index] < bb[o_index + 4]));
			bb[p_index] = f ? bb[p_index] : bb[p_index + 4];
			bb[o_index] = f ? bb[o_index] : bb[o_index + 4];
		}
		{
			constexpr int i = 8;
			constexpr int p_index = i + 2;
			constexpr int o_index = i;

			const bool f = bool(bb[p_index] < bb[p_index + 4]) | (bool(bb[p_index] == bb[p_index + 4]) & bool(bb[o_index] < bb[o_index + 4]));
			bb[p_index] = f ? bb[p_index] : bb[p_index + 4];
			bb[o_index] = f ? bb[o_index] : bb[o_index + 4];
		}
	}
	{
		constexpr int i = 0;
		constexpr int p_index = i + 2;
		constexpr int o_index = i;

		const bool f = bool(bb[p_index] < bb[p_index + 8]) | (bool(bb[p_index] == bb[p_index + 8]) & bool(bb[o_index] < bb[o_index + 8]));
		P_dest = f ? bb[p_index] : bb[p_index + 8];
		O_dest = f ? bb[o_index] : bb[o_index + 8];
	}
}

std::array<uint64_t, 2>board_unique(const uint64_t original_player, const uint64_t original_opponent) {
	std::array<uint64_t, 2> answer;
	board_unique(original_player, original_opponent, answer[0], answer[1]);
	return answer;
}
std::array<uint64_t, 2>board_unique(const std::array<uint64_t, 2> original_bitboards) {
	std::array<uint64_t, 2> answer;
	board_unique(original_bitboards[0], original_bitboards[1], answer[0], answer[1]);
	return answer;
}

uint64_t get_some_moves(const uint64_t player, const uint64_t mask, const int dir)
{
	uint64_t flip_l, flip_r;
	uint64_t mask_l, mask_r;
	const int dir2 = dir + dir;

	flip_l = mask & (player << dir);      flip_r = mask & (player >> dir);
	flip_l |= mask & (flip_l << dir);     flip_r |= mask & (flip_r >> dir);
	mask_l = mask & (mask << dir);        mask_r = mask & (mask >> dir);
	flip_l |= mask_l & (flip_l << dir2);  flip_r |= mask_r & (flip_r >> dir2);
	flip_l |= mask_l & (flip_l << dir2);  flip_r |= mask_r & (flip_r >> dir2);

	return (flip_l << dir) | (flip_r >> dir);
}
uint64_t get_moves(const uint64_t player, const uint64_t opponent)
{
	const uint64_t mask = opponent & 0x7E7E7E7E7E7E7E7Eull;

	return (get_some_moves(player, mask, 1)
		| get_some_moves(player, opponent, 8)
		| get_some_moves(player, mask, 7)
		| get_some_moves(player, mask, 9))
		& ~(player | opponent);
}

int bit_count(const uint64_t b)
{
	uint64_t c = b
		- ((b >> 1) & 0x7777777777777777ULL)
		- ((b >> 2) & 0x3333333333333333ULL)
		- ((b >> 3) & 0x1111111111111111ULL);
	c = ((c + (c >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL;

	return  (int)(c >> 56);
}

__m256i flipVertical(const __m256i dbd) {
	const __m256i byteswap_table = _mm256_set_epi8(
		8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7
	);
	return _mm256_shuffle_epi8(dbd, byteswap_table);
}
__m256i upperbit(__m256i p) {
	p = _mm256_or_si256(p, _mm256_srli_epi64(p, 1));
	p = _mm256_or_si256(p, _mm256_srli_epi64(p, 2));
	p = _mm256_or_si256(p, _mm256_srli_epi64(p, 4));
	p = _mm256_andnot_si256(_mm256_srli_epi64(p, 1), p);
	p = flipVertical(p);
	p = _mm256_and_si256(p, _mm256_sub_epi64(_mm256_setzero_si256(), p));
	return flipVertical(p);
}
uint64_t hor(const __m256i lhs) {
	__m128i lhs_xz_yw = _mm_or_si128(_mm256_castsi256_si128(lhs), _mm256_extractf128_si256(lhs, 1));
	return uint64_t(_mm_cvtsi128_si64(lhs_xz_yw) | _mm_extract_epi64(lhs_xz_yw, 1));
}
uint64_t flip(const uint64_t player, const uint64_t opponent, const int pos) {
	const __m256i pppp = _mm256_set1_epi64x(player);
	const __m256i oooo = _mm256_set1_epi64x(opponent);
	const __m256i OM = _mm256_and_si256(oooo, _mm256_set_epi64x(
		0xFFFFFFFFFFFFFFFFULL, 0x7E7E7E7E7E7E7E7EULL, 0x7E7E7E7E7E7E7E7EULL, 0x7E7E7E7E7E7E7E7EULL));
	__m256i flipped, outflank, mask;
	mask = _mm256_srli_epi64(_mm256_set_epi64x(
		0x0080808080808080ULL, 0x7F00000000000000ULL, 0x0102040810204000ULL, 0x0040201008040201ULL), 63 - pos);
	outflank = _mm256_and_si256(upperbit(_mm256_andnot_si256(OM, mask)), pppp);
	flipped = _mm256_and_si256(_mm256_slli_epi64(_mm256_sub_epi64(_mm256_setzero_si256(), outflank), 1), mask);
	mask = _mm256_slli_epi64(_mm256_set_epi64x(
		0x0101010101010100ULL, 0x00000000000000FEULL, 0x0002040810204080ULL, 0x8040201008040200ULL), pos);
	outflank = _mm256_and_si256(_mm256_and_si256(mask, _mm256_add_epi64(_mm256_or_si256(
		OM, _mm256_andnot_si256(mask, _mm256_set1_epi8(0xFF))), _mm256_set1_epi64x(1))), pppp);
	flipped = _mm256_or_si256(flipped, _mm256_and_si256(_mm256_sub_epi64(outflank, _mm256_add_epi64(
		_mm256_cmpeq_epi64(outflank, _mm256_setzero_si256()), _mm256_set1_epi64x(1))), mask));
	return hor(flipped);
}

uint64_t flip_slow(const uint64_t player, const uint64_t opponent, const int pos) {
	const int dir[8] = { -9,-8,-7,-1,1,7,8,9 };
	const uint64_t edge[8] = {
		0x01010101010101ffull,
		0x00000000000000ffull,
		0x80808080808080ffull,
		0x0101010101010101ull,
		0x8080808080808080ull,
		0xff01010101010101ull,
		0xff00000000000000ull,
		0xff80808080808080ull
	};
	uint64_t flipped = 0;
	for (int d = 0; d < 8; ++d) {
		if (((1ULL << pos) & edge[d]) == 0) {
			uint64_t f = 0;
			int x = pos + dir[d];
			for (; (opponent & (1ULL << x)) && ((1ULL << x) & edge[d]) == 0; x += dir[d]) {
				f |= (1ULL << x);
			}
			if (player & (1ULL << x)) flipped |= f;
		}
	}
	return flipped;
}

void test(const int seed, const int length) {
	std::mt19937_64 rnd(seed);
	const uint64_t initial_occupied = 0x0000'0018'1800'0000ULL;

	uint64_t count_testnum_flip_nonzero = 0;
	uint64_t count_testnum_flip_zero = 0;

	for (int iteration = 0; iteration < length; ++iteration) {
		uint64_t player = 0, opponent = 0;
		for (uint64_t i = 1; i; i <<= 1) {
			switch (std::uniform_int_distribution<uint64_t>(0, (i & initial_occupied) ? 1 : (2 + (iteration % 2)))(rnd)) {
			case 0:
				player |= i;
				break;
			case 1:
				opponent |= i;
				break;
			}
		}
		const uint64_t empty = ~(player | opponent);
		const uint64_t move_bb = get_moves(player, opponent);
		assert((empty | move_bb) == empty);
		assert(bit_count(move_bb) == _mm_popcnt_u64(move_bb));
		for (int pos = 0; pos < 64; ++pos)if ((1ULL << pos) & empty) {
			const uint64_t flip1 = flip(player, opponent, pos);
			const uint64_t flip2 = flip_slow(player, opponent, pos);
			assert(flip1 == flip2);
			assert((flip1 != 0) == (((1ULL << pos) & move_bb) != 0));
			if (flip1 == 0) {
				++count_testnum_flip_zero;
			}
			else {
				++count_testnum_flip_nonzero;
			}
		}
	}
	std::cout << "test succeeded" << std::endl;
	std::cout << "count_testnum_flip_nonzero = " << count_testnum_flip_nonzero << std::endl;
	std::cout << "count_testnum_flip_zero = " << count_testnum_flip_zero << std::endl;
}

inline uint32_t bitscan_forward64(const uint64_t x, uint32_t *dest) {


#ifdef _MSC_VER
	return _BitScanForward64(reinterpret_cast<unsigned long *>(dest), x);
#else
	return x ? *dest = __builtin_ctzll(x), 1 : 0;
#endif

}

constexpr uint64_t MASK_8x8to6x6 = ~0xFF81'8181'8181'81FFULL;
const auto codebook = "56789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#_";
uint8_t inverse_codebook[128];

uint8_t ternary_encoding_table[32][32];
uint8_t ternary_decoding_table[243][2];
uint8_t ternary_encoding_string_table[16][16];
uint8_t ternary_decoding_string_table[81][2];
void init_ternary_tables() {
	for (int i = 0; i < 243; ++i) {
		int d[5] = {}, b[3] = {};
		for (int n = i, j = 0; j < 5; ++j, n /= 3) {
			d[j] = n % 3;
			b[d[j]] |= 1 << j;
		}
		assert((b[1] & b[2]) == 0);
		ternary_encoding_table[b[1]][b[2]] = i;
		ternary_decoding_table[i][0] = b[1];
		ternary_decoding_table[i][1] = b[2];
	}
	for (int i = 0; i < 81; ++i) {
		int d[4] = {}, b[3] = {};
		for (int n = i, j = 0; j < 4; ++j, n /= 3) {
			d[j] = n % 3;
			b[d[j]] |= 1 << j;
		}
		assert((b[1] & b[2]) == 0);
		ternary_encoding_string_table[b[1]][b[2]] = i;
		ternary_decoding_string_table[i][0] = b[1];
		ternary_decoding_string_table[i][1] = b[2];
	}
	for (int i = 0; i < 127; ++i)inverse_codebook[i] = 127;
	for (int i = 0; i < 81; ++i) {
		inverse_codebook[codebook[i]] = i;
	}
}
uint64_t encode_bb(const std::array<uint64_t, 2> bb) {
	uint64_t bb0 = _pext_u64(bb[0], MASK_8x8to6x6);
	uint64_t bb1 = _pext_u64(bb[1], MASK_8x8to6x6);
	uint64_t answer = 0;
	for (int i = 0; i < 36; i += 5) {
		const uint64_t index0 = bb0 % 32;
		const uint64_t index1 = bb1 % 32;
		assert((index0 & index1) == 0);
		answer = answer * 256 + ternary_encoding_table[index0][index1];
		bb0 /= 32;
		bb1 /= 32;
	}
	return answer;
}
std::array<uint64_t, 2> decode_bb(uint64_t code) {
	uint64_t bb0 = 0, bb1 = 0;
	for (int i = 0; i < 64; i += 8) {
		const uint64_t index = code % 256;
		bb0 = bb0 * 32 + ternary_decoding_table[index][0];
		bb1 = bb1 * 32 + ternary_decoding_table[index][1];
		code /= 256;
	}
	const uint64_t answer0 = _pdep_u64(bb0, MASK_8x8to6x6);
	const uint64_t answer1 = _pdep_u64(bb1, MASK_8x8to6x6);
	return std::array<uint64_t, 2>{answer0, answer1};
}

std::array<char, 9> encode_string_bb(const std::array<uint64_t, 2> bb) {
	uint64_t bb0 = _pext_u64(bb[0], MASK_8x8to6x6);
	uint64_t bb1 = _pext_u64(bb[1], MASK_8x8to6x6);
	std::array<char, 9> answer;
	for (int i = 0; i < 9; ++i) {
		const uint64_t index0 = bb0 % 16;
		const uint64_t index1 = bb1 % 16;
		assert((index0 & index1) == 0); 
		
		answer[i] = codebook[ternary_encoding_string_table[index0][index1]];
		bb0 /= 16;
		bb1 /= 16;
	}
	return answer;
}
std::array<uint64_t, 2> decode_string_bb(std::array<char, 9> code) {
	uint64_t bb0 = 0, bb1 = 0;
	for (int i = 8; 0 <= i; --i) {
		const auto index = inverse_codebook[code[i]];
		assert(index < 81);
		bb0 = bb0 * 16 + ternary_decoding_string_table[index][0];
		bb1 = bb1 * 16 + ternary_decoding_string_table[index][1];
	}
	const uint64_t answer0 = _pdep_u64(bb0, MASK_8x8to6x6);
	const uint64_t answer1 = _pdep_u64(bb1, MASK_8x8to6x6);
	return std::array<uint64_t, 2>{answer0, answer1};
}

uint64_t shift_and_xor_forward(const uint64_t x, const int width)
{
	assert(-63 <= width && width <= 63 && width != 0);
	const uint64_t a = width > 0 ? x << width : x >> -width;
	return a ^ x;
}
uint64_t shift_and_xor_backward(const uint64_t x, const int width)
{
	assert(-63 <= width && width <= 63 && width != 0);
	uint64_t answer = 0;
	if (width > 0) {
		const uint64_t base_mask = ((1ULL << width) - 1ULL);
		answer = x & base_mask;
		for (int i = width; i < 64; i += width) {
			const uint64_t mask = base_mask << i;
			answer |= ((answer << width) ^ x) & mask;
		}
		return answer;
	}
	else if (width < 0) {
		const uint64_t base_mask = ((1ULL << -width) - 1ULL) << (64 + width);
		answer = x & base_mask;
		for (int i = -width; i < 64; i -= width) {
			const uint64_t mask = base_mask >> i;
			answer |= ((answer >> -width) ^ x) & mask;
		}
		return answer;
	}
	else assert(false);
}
uint64_t xorshift64plus(const uint64_t a)
{
	const uint64_t b = a * 13499267949257065399ULL;
	const uint64_t c = shift_and_xor_forward(b, 13);
	const uint64_t d = shift_and_xor_forward(c, -7);
	const uint64_t e = shift_and_xor_forward(d, 17);
	const uint64_t f = e * 3767363990679839801ULL;
	return f;
}
uint64_t xorshift64plus_inverse(uint64_t f) {
	const uint64_t e = f * 1000000009ULL;
	const uint64_t d = shift_and_xor_backward(e, 17);
	const uint64_t c = shift_and_xor_backward(d, -7);
	const uint64_t b = shift_and_xor_backward(c, 13);
	const uint64_t a = b * 1000000007ULL;
	return a;
}


enum {
	A1, B1, C1, D1, E1, F1, G1, H1,
	A2, B2, C2, D2, E2, F2, G2, H2,
	A3, B3, C3, D3, E3, F3, G3, H3,
	A4, B4, C4, D4, E4, F4, G4, H4,
	A5, B5, C5, D5, E5, F5, G5, H5,
	A6, B6, C6, D6, E6, F6, G6, H6,
	A7, B7, C7, D7, E7, F7, G7, H7,
	A8, B8, C8, D8, E8, F8, G8, H8
};


constexpr int BUFSIZE = 2 * 1024 * 1024;


template<int N>class BufferedReader {
private:
	std::ifstream reading_file;

	std::vector<std::array<char, N>> buffer_lines;
	int64_t filenumber;
	std::string filename_;
	int64_t buffer_lines_size;
	int64_t buffer_lines_index;
	bool end_flag;

	bool change_file() {
		reading_file.close();
		if (filenumber == -1) {
			return false;
		} 
		const std::string next_filename = filename_ + "_" + std::to_string(++filenumber) + ".txt";
		std::error_code ec;
		if (std::filesystem::exists(next_filename, ec)){
			reading_file.open(next_filename, std::ios::in);
			if (reading_file.fail()){
				std::cout << "BufferedReader change_file failed. filename=" << next_filename << std::endl;
				assert(false);
			}
			return true;
		}
		return false;
	}

	void read_lines() {
		std::string line_getline;
		if (end_flag)return;
		buffer_lines_index = 0;
		int64_t i = 0;
		for (; i < buffer_lines_size; ++i) {
			if (!std::getline(reading_file, line_getline)){
				if (!change_file())break;
				if (!std::getline(reading_file, line_getline))break;
			}
			for (int j = 0; j < N; ++j)buffer_lines[i][j] = line_getline[j];
		}
		if (i < buffer_lines_size){
			end_flag = true;
			buffer_lines_size = i;
		}
	}

public:

	BufferedReader(const std::string filename, const int bufsize = BUFSIZE) {
		std::error_code ec;
		if (std::filesystem::exists(filename, ec)) {
			reading_file.open(filename, std::ios::in);
			if (reading_file.fail()){
				std::cout << "BufferedReader 1 failed. filename=" << filename << std::endl;
				assert(false);
			}
			filenumber = -1;
		}
		else {
			if (std::filesystem::exists(filename + "_0.txt", ec)) {
				reading_file.open(filename + "_0.txt", std::ios::in);
				if (reading_file.fail()){
					std::cout << "BufferedReader 2 failed. filename=" << (filename + "_0.txt") << std::endl;
					assert(false);
				}
				filenumber = 0;
			}
			else assert(false);
		}
		filename_ = filename;
		buffer_lines_size = bufsize;
		buffer_lines.resize(buffer_lines_size);
		buffer_lines_index = 0;
		end_flag = false;
		read_lines();
	}

	bool get_line(std::array<char, N> &answer) {
		if (buffer_lines_index == buffer_lines_size)read_lines();
		if (buffer_lines_index == buffer_lines_size)return false;
		answer = buffer_lines[buffer_lines_index++];
		return true;
	}
};

template<int N>class BufferedWriter {
private:
	std::ofstream writing_file;

	std::vector<char> buffer_lines;
	int64_t filenumber;
	std::string filename_;
	int64_t buffer_lines_size;
	int64_t buffer_lines_index;

	std::vector<std::string>output_filenames;
	bool is_empty;

	const int64_t FILESIZE_THRESHOLD = 20LL * 1000LL * 1000LL * 1000LL;

	void check_and_change_file() {
		if (filenumber == -1) return;
		std::error_code ec;
		if (!std::filesystem::exists(output_filenames.back(), ec)){
			std::cout << "error: file doesnot exist: filename=" << output_filenames.back() << std::endl;
			assert(false);
		}
		if (std::filesystem::file_size(output_filenames.back()) >= FILESIZE_THRESHOLD) {
			const std::string next_filename = filename_ + "_" + std::to_string(++filenumber) + ".txt";
			writing_file.flush();
			writing_file.close();
			writing_file.open(next_filename, std::ios::out);
			if (writing_file.fail()){
				std::cout << "BufferedWriter check_and_change_file open failed. filename=" << next_filename << std::endl;
				assert(false);
			}
			output_filenames.push_back(next_filename);
		}
	}

	void write_lines() {
		check_and_change_file();
		buffer_lines[(N + 1) * buffer_lines_index] = '\0';
		writing_file << buffer_lines.data();
		writing_file.flush();
	}

public:

	BufferedWriter(const std::string filename, const bool renban = false, const int bufsize = 20 * 1000 * 1000) {
		is_empty = false;
		if (renban) {
			writing_file.open(filename + "_0.txt", std::ios::out);
			if (writing_file.fail()){
				std::cout << "BufferedWriter renban==true open failed. filename=" << (filename + "_0.txt") << std::endl;
				assert(false);
			}
			output_filenames.push_back(filename + "_0.txt");
			filename_ = filename;
			filenumber = 0;
		}
		else {
			writing_file.open(filename, std::ios::out);
			if (writing_file.fail()){
				std::cout << "BufferedWriter renban==false open failed. filename=" << filename << std::endl;
				assert(false);
			}
			output_filenames.push_back(filename);
			filename_ = "";
			filenumber = -1;
		}
		buffer_lines_index = 0;
		buffer_lines_size = bufsize;
		buffer_lines.resize((N + 1) * buffer_lines_size + 1);
	}

	BufferedWriter() {
		is_empty = true;
	}

	void put_line(const std::array<char, N> &line) {
		assert(!is_empty);
		for (int i = 0; i < N; ++i) {
			buffer_lines[buffer_lines_index * (N + 1) + i] = line[i];
		}
		buffer_lines[buffer_lines_index * (N + 1) + N] = '\n';
		++buffer_lines_index;
		if (buffer_lines_index == buffer_lines_size) {
			write_lines();
			buffer_lines_index = 0;
		}
	}

	void flush() {
		assert(!is_empty);
		write_lines();
		buffer_lines_index = 0;
	}

	void put_bulk(const std::vector<char> &lines){
		assert(!is_empty);
		write_lines();
		buffer_lines_index = 0;
		writing_file << lines.data();
		writing_file.flush();
	}

	std::vector<std::string> get_filenames() {
		assert(!is_empty);
		return output_filenames;
	}
};


int64_t get_sum_filesize(const std::string filename) {

	std::error_code ec;
	if (std::filesystem::exists(filename, ec)) {
		const int64_t filesize = std::filesystem::file_size(filename);
		return filesize;
	}
	int64_t sum_filesize = 0;
	for (int64_t i = 0;; ++i) {
		const std::string filename_ = filename + "_" + std::to_string(i) + ".txt";
		if (std::filesystem::exists(filename_, ec)) {
			sum_filesize += std::filesystem::file_size(filename_);
		}
		else break;
	}
	return sum_filesize;
}



int32_t ComputeFinalScore(const uint64_t player, const uint64_t opponent) {


	const int32_t n_discs_p = _mm_popcnt_u64(player);
	const int32_t n_discs_o = _mm_popcnt_u64(opponent);

	int32_t score = n_discs_p - n_discs_o;

	if (score < 0) score -= 36 - n_discs_p - n_discs_o;
	else if (score > 0) score += 36 - n_discs_p - n_discs_o;

	return score;
}

char encode_score(const int32_t score) {
	assert(-36 <= score && score <= 36 && std::abs(score) % 2 == 0);
	const auto s = (score + 36) / 2;
	return codebook[s];
}
int32_t decode_score(const char code) {
	const auto s = inverse_codebook[code];
	return (s * 2) - 36;
}

void forward(const int N, const std::array<char, 9> &task, std::vector<uint64_t>&results) {
	const std::array<uint64_t, 2> bb = decode_string_bb(task);
	uint64_t bb_moves = get_moves(bb[0], bb[1]) & MASK_8x8to6x6;
	for (uint32_t square = 0; bitscan_forward64(bb_moves, &square); bb_moves &= bb_moves - 1) {
		const uint64_t bb_flip = flip(bb[0], bb[1], square);
		assert(bb_flip);
		uint64_t next_p = bb[1] ^ bb_flip;
		uint64_t next_o = bb[0] ^ (bb_flip | (1ULL << square));
		assert((next_p & next_o) == 0);
		if (N < 36) {
			const uint64_t next_bb_moves = get_moves(next_p, next_o) & MASK_8x8to6x6;
			if (next_bb_moves == 0) {
				uint64_t tmp = next_p;
				next_p = next_o;
				next_o = tmp;
			}
			if ((get_moves(next_p, next_o) & MASK_8x8to6x6) == 0)continue;
		}
		assert(_mm_popcnt_u64(next_p | next_o) == N);
		uint64_t unique_p = 0, unique_o = 0;
		board_unique(next_p, next_o, unique_p, unique_o);
		results.push_back(encode_bb(std::array<uint64_t, 2>{ unique_p, unique_o }));
	}
}

void firststep_enumerate_legalmoves(const int N, const std::string reading_filename, const std::string writing_filename) {
	
		//const std::string reading_filename = "forward_" + zerofill_itos(N - 1) + "_boardlist";
		//const std::string writing_filename = "forward_" + zerofill_itos(N - 0) + "_boardlist-un-sort-uniquefy";

	const auto reading_filesize = get_sum_filesize(reading_filename);
	assert(reading_filesize % 10 == 0);
	const uint64_t reading_linesize = reading_filesize / 10;

	BufferedReader<9>reading_file(reading_filename);
	BufferedWriter<9>writing_first_temp_file(writing_filename, true);

	constexpr int64_t CHUNKSIZE = 10'000'000;

#pragma omp parallel for schedule(dynamic)
	for (int64_t i = 0; i < reading_linesize; i += CHUNKSIZE) {
		std::vector<std::array<char, 9>> tasks;
#pragma omp critical(critical_reading_file)
		{
			std::array<char, 9> line;
			for (uint64_t j = 0; j < CHUNKSIZE; ++j) {
				if (!reading_file.get_line(line))break;
				tasks.push_back(line);
			}
		}
		if (tasks.size() == 0)continue;
		std::vector<uint64_t>results;
		for (const auto task : tasks) {
			forward(N, task, results);
		}
		if (results.size() == 0)continue;
		std::sort(results.begin(), results.end());
		results.erase(std::unique(results.begin(), results.end()), results.end());

		std::vector<char> output_tmp_string(10 * results.size() + 1);
		for (int64_t j = 0; j < results.size(); ++j) {
			const std::array<char, 9> a = encode_string_bb(decode_bb(results[j]));
			for (int k = 0; k < 9; ++k) {
				output_tmp_string[j * 10 + k] = a[k];
			}
			output_tmp_string[j * 10 + 9] = '\n';
		}
		output_tmp_string[results.size() * 10] = '\0';
#pragma omp critical(critical_writing_file)
		{
			writing_first_temp_file.put_bulk(output_tmp_string);
		}
	}
	writing_first_temp_file.flush();
}
void firststep_enumerate_legalmoves_countonly(const int N, const std::string reading_filename) {
	
		//const std::string reading_filename = "forward_" + zerofill_itos(N - 1) + "_boardlist";

	const auto reading_filesize = get_sum_filesize(reading_filename);
	assert(reading_filesize % 10 == 0);
	const uint64_t reading_linesize = reading_filesize / 10;

	BufferedReader<9>reading_file(reading_filename);

	constexpr int64_t CHUNKSIZE = 10'000'000;

	uint64_t linecount = 0;

#pragma omp parallel for schedule(dynamic)
	for (int64_t i = 0; i < reading_linesize; i += CHUNKSIZE) {
		std::vector<std::array<char, 9>> tasks;
#pragma omp critical(critical_reading_file)
		{
			std::array<char, 9> line;
			for (uint64_t j = 0; j < CHUNKSIZE; ++j) {
				if (!reading_file.get_line(line))break;
				tasks.push_back(line);
			}
		}
		if (tasks.size() == 0)continue;
		std::vector<uint64_t>results;
		for (const auto task : tasks) {
			forward(N, task, results);
		}
		if (results.size() == 0)continue;
		std::sort(results.begin(), results.end());
		results.erase(std::unique(results.begin(), results.end()), results.end());

#pragma omp critical(critical_writing_file)
		{
			linecount += results.size();
		}
	}
	std::cout << "firststep_enumerate_legalmoves_countonly " << "N=" << N << ", linecount=" << linecount << std::endl;
}

enum{
	CHUNKDATA_INDEX_LOWERBOUND = 0,
	CHUNKDATA_INDEX_UPPERBOUND = 1,
	CHUNKDATA_VALUE_LOWERBOUND = 2,
	CHUNKDATA_VALUE_UPPERBOUND = 3
};
const int64_t LINE_CHUNK_SIZE = 50000;

std::vector<std::array<uint64_t, 4>> descending_sort_singlefile_and_get_chunkdata(const std::string filename) {

	std::error_code ec;
	if (!std::filesystem::exists(filename, ec)){
		std::cout << "descending_sort_singlefile_and_get_chunkdata file doesnot exist. filename=" << filename << std::endl;
		assert(false);
	}
	const int64_t filesize = std::filesystem::file_size(filename);
	assert(filesize % 10 == 0);
	const int64_t linesize = filesize / 10;
	if(filesize == 0){
		std::cout << "warning: descending_sort_singlefile_and_get_chunkdata file filesize==0. filename=" << filename << std::endl;
		assert(false);
	}

	std::vector<char> tmp_string(filesize + 1);
	std::ifstream reading_file;
	reading_file.open(filename, std::ios::in);
	if (reading_file.fail()){
		std::cout << "descending_sort_singlefile_and_get_chunkdata reading_file open failed. filename=" << filename << std::endl;
		assert(false);
	}
	reading_file.read(tmp_string.data(), filesize);
	reading_file.close();

	std::vector<uint64_t> codes(linesize);
#pragma omp parallel for schedule(static)
	for(int64_t j = 0; j < linesize; ++j){
		std::array<char, 9> line;
		for(int k = 0; k < 9; ++k){
			line[k] = tmp_string[j * 10 + k];
		}
		assert(tmp_string[j * 10 + 9] == '\n');
		codes[j] = -encode_bb(decode_string_bb(line));
	}

	__gnu_parallel::sort(codes.begin(), codes.end());
	codes.erase(std::unique(codes.begin(), codes.end()), codes.end());

	const int64_t lines_unique_size = codes.size();
	tmp_string.resize(lines_unique_size * 10 + 1);
	tmp_string[lines_unique_size * 10] = '\0';

#pragma omp parallel for schedule(static)
	for(int64_t j = 0; j < lines_unique_size; ++j){
		std::array<char, 9> line = encode_string_bb(decode_bb(-codes[j]));
		for(int k = 0; k < 9; ++k){
			tmp_string[j * 10 + k] = line[k];
		}
		tmp_string[j * 10 + 9] = '\n';
	}

	std::vector<std::array<uint64_t, 4>> chunkdata;
	for(int64_t j = 0; j < lines_unique_size;){
		std::array<uint64_t, 4> chunkdata_element;
		std::array<char, 9> line;
		for(int k = 0; k < 9; ++k){
			line[k] = tmp_string[j * 10 + k];
		}
		chunkdata_element[CHUNKDATA_VALUE_UPPERBOUND] = encode_bb(decode_string_bb(line));
		chunkdata_element[CHUNKDATA_INDEX_LOWERBOUND] = j;
		int64_t rbegin = std::min(j + LINE_CHUNK_SIZE - 1LL, lines_unique_size - 1LL);
		assert(j < rbegin);
		if (j + LINE_CHUNK_SIZE == lines_unique_size - 1LL)rbegin = lines_unique_size - 1LL;
		for(int k = 0; k < 9; ++k){
			line[k] = tmp_string[rbegin * 10 + k];
		}
		chunkdata_element[CHUNKDATA_VALUE_LOWERBOUND] = encode_bb(decode_string_bb(line));
		chunkdata_element[CHUNKDATA_INDEX_UPPERBOUND] = rbegin;
		assert(chunkdata_element[CHUNKDATA_VALUE_LOWERBOUND] <= chunkdata_element[CHUNKDATA_VALUE_UPPERBOUND]);
		assert(chunkdata_element[CHUNKDATA_INDEX_LOWERBOUND] <= chunkdata_element[CHUNKDATA_INDEX_UPPERBOUND]);
		chunkdata.push_back(chunkdata_element);

		j = rbegin + 1;
	}

	for(int64_t j = 1; j < chunkdata.size(); ++j){
		assert(chunkdata[j - 1][CHUNKDATA_VALUE_LOWERBOUND] > chunkdata[j][CHUNKDATA_VALUE_UPPERBOUND]);
		assert(chunkdata[j - 1][CHUNKDATA_INDEX_UPPERBOUND] + 1 == chunkdata[j][CHUNKDATA_INDEX_LOWERBOUND]);
	}

	std::filesystem::remove(filename);
	std::ofstream writing_file;
	writing_file.open(filename, std::ios::out);
	if (writing_file.fail()){
		std::cout << "descending_sort_singlefile_and_get_chunkdata writing_file open failed. filename=" << filename << std::endl;
		assert(false);
	}
	writing_file << tmp_string.data() << std::flush;
	writing_file.close();

	return chunkdata;
}
/*
int64_t find_suitable_line_from_singlefile(const std::string filename, const uint64_t target_code, std::map<uint64_t, int64_t>& code_map){

	std::cout << "start find_suitable_line_from_singlefile. filename=" << filename << " target_code=" << target_code << std::endl;

	const auto it1 = code_map.rbegin();
	const uint64_t max_key_code = it1->first;
	assert(it1->second == 0);
	const auto it2 = code_map.begin();
	const uint64_t min_key_code = it2->first;
	const int64_t min_value_index = it2->second;
	assert(min_value_index == 0 ? min_key_code == max_key_code : min_key_code < max_key_code);
	
	if(target_code >= max_key_code){
		std::cout << "end find_suitable_line_from_singlefile. filename=" << filename << " target_code=" << target_code << " max_key_code=" << max_key_code << std::endl;	
		return 0;
	}
	if(target_code == min_key_code){
		std::cout << "end find_suitable_line_from_singlefile. filename=" << filename << " target_code=" << target_code << " min_key_code=" << min_key_code << std::endl;
		return min_value_index;
	}
	if(target_code < min_key_code){
		std::cout << "end find_suitable_line_from_singlefile. filename=" << filename << " target_code=" << target_code << " min_key_code=" << min_key_code << std::endl;
		return std::numeric_limits<int64_t>::max();
	}

	const auto it3 = code_map.find(target_code);
	if(it3 != code_map.end()){
		std::cout << "end find_suitable_line_from_singlefile. filename=" << filename << " target_code=" << target_code << " it3->second=" << it3->second << std::endl;
		return it3->second;
	}

	std::error_code ec;
	assert(std::filesystem::exists(filename, ec));
	std::ifstream reading_file;
	reading_file.open(filename, std::ios::in);
	if (reading_file.fail()){
		std::cout << "find_suitable_line_from_singlefile reading_file open failed. filename=" << filename << std::endl;
		assert(false);
	}

	const auto it4 = code_map.upper_bound(target_code);
	const auto it5 = std::prev(it4);
	assert(it4 != code_map.end());
	assert(it5 != code_map.end());
	int64_t lb_index = it4->second, ub_index = it5->second;
	assert(0 <= lb_index && lb_index < ub_index && ub_index <= min_value_index);
	uint64_t lb_index_bigger_code = it4->first, ub_index_smaller_code = it5->first;
	assert(lb_index_bigger_code > target_code && target_code >= ub_index_smaller_code);

	while(lb_index + 1 < ub_index){
		const int64_t mid_index = lb_index / 2  + ub_index / 2 + (lb_index % 2 + ub_index % 2) / 2;
		reading_file.seekg(mid_index * 10, std::ios::beg);
		std::string line;
		std::getline(reading_file, line);
		std::array<char, 9> array_line;
		for(int j = 0; j < 9; ++j)array_line[j] = line[j];
		assert(line[9] == '\n' || line[9] == '\0');
		const uint64_t mid_code = encode_bb(decode_string_bb(array_line));
		code_map[mid_code] = mid_index;
		if(mid_code == target_code){
			reading_file.close();
			std::cout << "end find_suitable_line_from_singlefile. filename=" << filename << " target_code=" << target_code << " mid_index=" << mid_index << std::endl;
			return mid_index;
		}
		if(target_code >= mid_code){
			ub_index = mid_index;
			ub_index_smaller_code = mid_code;
		}
		else{
			lb_index = mid_index;
			lb_index_bigger_code = mid_code;
		}
		assert(lb_index_bigger_code > target_code && target_code >= ub_index_smaller_code);
	}
	reading_file.close();

	std::cout << "end find_suitable_line_from_singlefile. filename=" << filename << " target_code=" << target_code << " lb_index=" << lb_index << " ub_index=" << ub_index << std::endl;

	return ub_index;
}

void compute_range_from_code_maps(const std::vector<std::map<uint64_t, int64_t>> &code_maps, const uint64_t target_code, int64_t &lowerbound_remaining_lines, int64_t &upperbound_remaining_lines) {

	std::cout << "start compute_range_from_code_maps. target_code=" << target_code << std::endl;

	lowerbound_remaining_lines = 0;
	upperbound_remaining_lines = 0;

	for(int64_t i = 0; i < code_maps.size(); ++i){

		const auto it1 = code_maps[i].rbegin();
		const uint64_t max_key_code = it1->first;
		if(max_key_code <= target_code){
			continue;
		}

		const auto it2 = code_maps[i].begin();
		const uint64_t min_key_code = it2->first;
		if(min_key_code == target_code){
			lowerbound_remaining_lines += it2->second;
			upperbound_remaining_lines += it2->second;
			continue;
		}
		if(min_key_code > target_code){
			lowerbound_remaining_lines += it2->second + 1;
			upperbound_remaining_lines += it2->second + 1;
			continue;
		}

		const auto it3 = code_maps[i].upper_bound(target_code);
		assert(it3 != code_maps[i].end());
		const auto it4 = std::prev(it3);
		assert(it4 != code_maps[i].end());
		assert(it4->first <= target_code && target_code < it3->first);
		assert(it4->second > it3->second);
		if(it4->first == target_code){
			lowerbound_remaining_lines += it4->second;
			upperbound_remaining_lines += it4->second;
			continue;
		}
		upperbound_remaining_lines += it4->second;
		lowerbound_remaining_lines += it3->second + 1;
	}

	std::cout << "end compute_range_from_code_maps. target_code=" << target_code << " lowerbound=" << lowerbound_remaining_lines << " upperbound=" << upperbound_remaining_lines << std::endl;

}

void get_linesizes(const std::string filename, std::vector<int64_t>&linesizes, int64_t &num_files,int64_t &sum_lines){
	linesizes.clear();
	num_files = 0;
	sum_lines = 0;
	for(;; ++num_files){
		const std::string filename_ = filename + "_" + std::to_string(num_files) + ".txt";
		std::error_code ec;
		if (!std::filesystem::exists(filename_, ec))break;
		const int64_t filesize_ = std::filesystem::file_size(filename_);
		assert(filesize_ % 10 == 0);
		linesizes.push_back(filesize_ / 10);
		sum_lines += filesize_ / 10;
	}
}

void get_mincode_maxcode_and_trim_codemap(
	const std::string filename,
	const std::vector<int64_t>&linesizes,
	const int64_t num_files,
	const int64_t sum_lines,
	std::vector<std::map<uint64_t, int64_t>> &code_maps,
	uint64_t &min_code,
	uint64_t &max_code){

	min_code = 0xFFFF'FFFF'FFFF'FFFFULL;
	max_code = 0;
	for(int64_t i = 0; i < num_files; ++i){
		if(linesizes[i] == 0)continue;

		const auto it1 = code_maps[i].rbegin();
		assert(it1->second == 0);
		max_code = std::max(max_code, it1->first);

		const std::string filename_ = filename + "_" + std::to_string(i) + ".txt";
		std::ifstream reading_file;
		reading_file.open(filename_, std::ios::in);
		if (reading_file.fail()){
			std::cout << "find_suitable_line_from_ascending_renbanfile reading_file open failed. filename=" << filename_ << std::endl;
			assert(false);
		}
		reading_file.seekg(-10, std::ios::end);
		std::string line;
		std::getline(reading_file, line);
		std::array<char, 9> array_line;
		reading_file.close();
		for(int j = 0; j < 9; ++j)array_line[j] = line[j];
		assert(line[9] == '\n' || line[9] == '\0');
		const uint64_t code = encode_bb(decode_string_bb(array_line));
		min_code = std::min(min_code, code);
		if (code_maps[i].find(code) == code_maps[i].end()){
			code_maps[i][code] = linesizes[i] - 1;
		}
		else assert(code_maps[i][code] == linesizes[i] - 1);

		code_maps[i].erase(code_maps[i].begin(), code_maps[i].find(code));
	}
}

std::vector<int64_t> find_suitable_lines_from_ascending_renbanfile(const std::string filename, std::vector<std::map<uint64_t, int64_t>> &code_maps, std::map<int64_t, uint64_t> &mid_history) {



	std::cout << "start find_suitable_lines_from_ascending_renbanfile. filename=" << filename << std::endl;

	std::vector<int64_t>linesizes;
	int64_t num_files = 0, sum_lines = 0;
	get_linesizes(filename, linesizes, num_files, sum_lines);
	if(sum_lines <= 2'000'000'000LL){
		return std::vector<int64_t>(num_files, 0);
	}

	uint64_t min_code = 0xFFFF'FFFF'FFFF'FFFFULL, max_code = 0;
	get_mincode_maxcode_and_trim_codemap(filename, linesizes, num_files, sum_lines, code_maps, min_code, max_code);




	auto it_history = mid_history.upper_bound(sum_lines - 1'000'000'000LL);
	if (it_history != mid_history.end()){
		if (it_history != mid_history.begin()){
			--it_history;
			if (it_history->first >= sum_lines - 2'000'000'000LL){
				int64_t sum_remaining_lines_mid = 0;
				std::vector<int64_t>answer;
				for(int64_t i = 0; i < num_files; ++i){
					if(linesizes[i] == 0){
						answer.push_back(std::numeric_limits<int64_t>::max());
						continue;
					}
					const std::string filename_ = filename + "_" + std::to_string(i) + ".txt";
					answer.push_back(find_suitable_line_from_singlefile(filename_, it_history->second, code_maps[i]));
					if(answer.back() == std::numeric_limits<int64_t>::max())continue;
					sum_remaining_lines_mid += answer.back();
				}
				assert(1'000'000'000LL <= sum_lines - sum_remaining_lines_mid && sum_lines - sum_remaining_lines_mid <=  2'000'000'000LL);

				std::cout << "end find_suitable_lines_from_ascending_renbanfile. filename=" << filename << " sum_lines=" << sum_lines << " sum_remaining_lines_mid=" << sum_remaining_lines_mid << " mid=" << it_history->second << std::endl;
				return answer;
			}
		}
	}
	
	for(int64_t iter = 0;; ++iter){
		const uint64_t mid = min_code / 2 + max_code / 2 + (min_code % 2 + max_code % 2) / 2;

		const int64_t lowerbound = 0, upperbound = 0;
		compute_range_from_code_maps(code_maps, mid, lowerbound, upperbound);
		std::cout << "mid=" << mid << " lowerbound=" << lowerbound << " upperbound=" << upperbound << "min_code=" << min_code << " max_code=" << max_code << std::endl;
		if(sum_lines - lowerbound < 1'000'000'000LL){
			min_code = mid;
			continue;
		}
		if(2'000'000'000LL < sum_lines - upperbound){
			max_code = mid;
			continue;
		}

		if(iter > 1000){
			std::cout << "iter > 1000" << std::endl;
			assert(false);
		}




		int64_t sum_remaining_lines_mid = 0;
		std::vector<int64_t>answer;
		for(int64_t i = 0; i < num_files; ++i){
			if(linesizes[i] == 0){
				answer.push_back(std::numeric_limits<int64_t>::max());
				continue;
			}
			const std::string filename_ = filename + "_" + std::to_string(i) + ".txt";
			answer.push_back(find_suitable_line_from_singlefile(filename_, mid, code_maps[i]));
			if(answer.back() == std::numeric_limits<int64_t>::max())continue;
			sum_remaining_lines_mid += answer.back();
		}

		mid_history[sum_remaining_lines_mid] = mid;

		if(sum_lines - sum_remaining_lines_mid < 1'000'000'000LL){
			min_code = mid;
		}
		else if(2'000'000'000LL < sum_lines - sum_remaining_lines_mid){
			max_code = mid;
		}
		else{

			std::cout << "end find_suitable_lines_from_ascending_renbanfile. filename=" << filename << " sum_lines=" << sum_lines << " sum_remaining_lines_mid=" << sum_remaining_lines_mid << " mid=" << mid << std::endl;
			return answer;
		}
	}
	assert(false);
	return std::vector<int64_t>(num_files, 0);
}


std::vector<int64_t> find_suitable_lines_from_ascending_renbanfile_upperbound(const std::string filename, std::vector<std::map<uint64_t, int64_t>> &code_maps, std::map<int64_t, uint64_t> &mid_history, const int64_t num_carry) {


	


	std::cout << "start find_suitable_lines_from_ascending_renbanfile_upperbound. filename=" << filename << std::endl;

	std::vector<int64_t>linesizes;
	int64_t num_files = 0, sum_lines = 0;
	get_linesizes(filename, linesizes, num_files, sum_lines);
	if(sum_lines <= 2'000'000'000LL - num_carry){
		return std::vector<int64_t>(num_files, 0);
	}


	uint64_t min_code = 0xFFFF'FFFF'FFFF'FFFFULL, max_code = 0;
	get_mincode_maxcode_and_trim_codemap(filename, linesizes, num_files, sum_lines, code_maps, min_code, max_code);



}
*/

std::vector<std::vector<std::array<uint64_t, 4>>> descending_sort_all_files_and_get_chunkdata(const int N, const std::string reading_filename, const std::string writing_filename, int64_t &num_files, std::vector<int64_t>&linesizes){
	
	assert(reading_filename != writing_filename);
	
	std::vector<std::vector<std::array<uint64_t, 4>>>chunk_data_vec;

	linesizes.clear();
	num_files = 0;
	for(;; ++num_files){
		const std::string filename = reading_filename + "_" + std::to_string(num_files) + ".txt";
		std::error_code ec;
		if (!std::filesystem::exists(filename, ec))break;
		chunk_data_vec.push_back(descending_sort_singlefile_and_get_chunkdata(filename));
		const int64_t filesize_ = std::filesystem::file_size(filename);
		assert(filesize_ % 10 == 0);
		linesizes.push_back(filesize_ / 10);
	}

	return chunk_data_vec;
}



void renbansort(const int N, const std::string reading_filename, const std::string writing_filename) {
	
	assert(reading_filename != writing_filename);

	int64_t num_files = 0;
	std::vector<int64_t>linesizes;
	std::vector<std::vector<std::array<uint64_t, 4>>>chunk_data_vec = descending_sort_all_files_and_get_chunkdata(N, reading_filename, writing_filename, num_files, linesizes);
	
	std::vector<uint64_t>chunk_min_values;
	for(int64_t i = 0; i < chunk_data_vec.size(); ++i){
		for(int64_t j = 0; j < chunk_data_vec[i].size(); ++j){
			chunk_min_values.push_back(chunk_data_vec[i][j][CHUNKDATA_VALUE_LOWERBOUND]);
		}
	}
	__gnu_parallel::sort(chunk_min_values.begin(), chunk_min_values.end());

	std::vector<int64_t>chunk_read_cursor(num_files, 0);
	for(int64_t i = 0; i < num_files; ++i){
		chunk_read_cursor[i] = chunk_data_vec[i].size();
	}

	int64_t chunk_min_values_cursor = 0;
	std::vector<uint64_t>codes_buf;
	codes_buf.reserve(2'100'000'000);

	BufferedWriter<9>writing_result_file(writing_filename, true);
	uint64_t write_max_value = 0;

	while(true){
		assert((2'000'000'000 - codes_buf.size()) / LINE_CHUNK_SIZE > 0);
		chunk_min_values_cursor += (2'000'000'000 - codes_buf.size() + LINE_CHUNK_SIZE - 1) / LINE_CHUNK_SIZE;
		chunk_min_values_cursor = std::min(chunk_min_values_cursor, (int64_t)chunk_min_values.size() - 1);

		const uint64_t read_threshold = chunk_min_values[chunk_min_values_cursor];
		int64_t read_count = 0;
		uint64_t write_threshold = 0xFFFF'FFFF'FFFF'FFFFULL;
		std::vector<int64_t>read_start_index(num_files, 0);
		for(int64_t i = 0; i < num_files; ++i){
			bool read_flag = false;
			while(chunk_read_cursor[i] > 0 && chunk_data_vec[i][chunk_read_cursor[i] - 1][CHUNKDATA_VALUE_LOWERBOUND] <= read_threshold){
				--chunk_read_cursor[i];
				++read_count;
				read_flag = true;
			}
			if(read_flag){
				read_start_index[i] = chunk_data_vec[i][chunk_read_cursor[i]][CHUNKDATA_INDEX_LOWERBOUND];
				if(chunk_read_cursor[i] > 0){
					write_threshold = std::min(write_threshold, chunk_data_vec[i][chunk_read_cursor[i] - 1][CHUNKDATA_VALUE_LOWERBOUND] - 1);
				}
			}
			else{
				if(chunk_read_cursor[i] == 0)assert(linesizes[i] == 0);
				else{
					assert(chunk_data_vec[i][chunk_read_cursor[i] - 1][CHUNKDATA_INDEX_UPPERBOUND] + 1 == linesizes[i]);
					write_threshold = std::min(write_threshold, chunk_data_vec[i][chunk_read_cursor[i] - 1][CHUNKDATA_VALUE_LOWERBOUND] - 1);
				}
				read_start_index[i] = linesizes[i];
			}
		}

		std::vector<int64_t>acc_num_lines(num_files + 1, 0);
		acc_num_lines[0] = codes_buf.size();
		for(int64_t i = 1; i <= num_files; ++i){
			acc_num_lines[i] = acc_num_lines[i - 1];
			assert(linesizes[i - 1] >= read_start_index[i - 1]);
			acc_num_lines[i] += linesizes[i - 1] - read_start_index[i - 1];
		}
		if(acc_num_lines.back() == 0){
			assert(read_count == 0);
			assert(write_threshold == 0xFFFF'FFFF'FFFF'FFFFULL);
			break;
		}
		codes_buf.resize(acc_num_lines.back());


#pragma omp parallel for schedule(dynamic)
		for(int64_t i = 0; i < num_files; ++i){
			if(read_start_index[i] == linesizes[i])continue;
			const int64_t num_pickup_lines = linesizes[i] - read_start_index[i];
			linesizes[i] = read_start_index[i];
			const std::string filename = reading_filename + "_" + std::to_string(i) + ".txt";
			std::vector<char>tmp_string(num_pickup_lines * 10 + 1);

#pragma omp critical(critical_reading_file)
			{
				std::ifstream reading_file;
				reading_file.open(filename, std::ios::in);
				if (reading_file.fail()){
					std::cout << "renbansort reading_file open failed. filename=" << filename << std::endl;
					assert(false);
				}
				reading_file.seekg(read_start_index[i] * 10, std::ios::beg);
				reading_file.read(tmp_string.data(), num_pickup_lines * 10);
				reading_file.close();
				std::filesystem::resize_file(filename, read_start_index[i] * 10);
			}

			std::array<char, 9> array_line;
			for(int64_t j = 0; j < num_pickup_lines; ++j){
				for(int k = 0; k < 9; ++k){
					array_line[k] = tmp_string[j * 10 + k];
				}
				assert(tmp_string[j * 10 + 9] == '\n' || tmp_string[j * 10 + 9] == '\0');
				codes_buf[acc_num_lines[i] + j] = encode_bb(decode_string_bb(array_line));
			}
		}


		__gnu_parallel::sort(codes_buf.begin(), codes_buf.end());
		codes_buf.erase(std::unique(codes_buf.begin(), codes_buf.end()), codes_buf.end());

		assert(codes_buf.size() > 0);
		assert(codes_buf[0] <= write_threshold);

		int64_t lb = 0, ub = codes_buf.size();
		while(lb + 1 < ub){
			const int64_t mid = lb / 2 + ub / 2 + (lb % 2 + ub % 2) / 2;
			if(codes_buf[mid] <= write_threshold){
				lb = mid;
			}
			else{
				ub = mid;
			}
		}
		assert(codes_buf[lb] <= write_threshold);
		if(ub < codes_buf.size())assert(write_threshold < codes_buf[ub]);

		const int64_t writing_lines_unique_size = ub;
		assert(write_max_value < codes_buf[0]);
		std::vector<std::array<char, 9>> result_lines(writing_lines_unique_size);

#pragma omp parallel for schedule(static)
		for(int64_t j = 0; j < writing_lines_unique_size; ++j){
			result_lines[j] = encode_string_bb(decode_bb(codes_buf[j]));
		}

		for(int64_t j = 0; j < writing_lines_unique_size; ++j){
			writing_result_file.put_line(result_lines[j]);
		}
		
		write_max_value = codes_buf[writing_lines_unique_size - 1];

		for(int64_t j = writing_lines_unique_size; j < codes_buf.size(); ++j){
			codes_buf[j - writing_lines_unique_size] = codes_buf[j];
		}
		codes_buf.resize(codes_buf.size() - writing_lines_unique_size);
	}

	writing_result_file.flush();


	for(int64_t i = 0; i < num_files; ++i){
		const std::string filename = reading_filename + "_" + std::to_string(i) + ".txt";
		std::error_code ec;
		assert(std::filesystem::exists(filename, ec));
		const int64_t filesize_ = std::filesystem::file_size(filename);
		assert(filesize_ == 0);
		std::filesystem::remove(filename);
	}

}


int solve(const int N, const std::string S) {

	const auto zerofill_itos = [](const int num) {
		if (num >= 10)return std::to_string(num);
		return std::string("0") + std::to_string(num);
	};


	if (N == 4 && S == "f") {
		std::ofstream writing_file;
		writing_file.open("forward_04_boardlist_0.txt", std::ios::out);
		const auto initial_board = std::array<uint64_t, 2>{ (1ULL << E4) | (1ULL << D5), (1ULL << D4) | (1ULL << E5) };
		const std::array<char, 9> p = encode_string_bb(initial_board);
		std::vector<char> writing_string;
		for( int i = 0; i < 9; ++i)writing_string.push_back(p[i]);
		writing_string.push_back('\n');
		writing_string.push_back('\0');
		writing_file << writing_string.data() << std::flush;
		writing_file.close();
		return 0;
	}
	else if(N == 5 && S == "f"){
		//555!5v555
		std::ofstream writing_file;
		writing_file.open("forward_05_boardlist_0.txt", std::ios::out);
		writing_file << "555!5v555\n" << std::flush;
		writing_file.close();
		return 0;
	}
	else if (6 <= N && N <= 36 && (S == "f")) {
		const std::string reading_filename = "forward_" + zerofill_itos(N - 1) + "_boardlist";
		const std::string writing_filename = "forward_" + zerofill_itos(N - 0) + "_boardlist_not_sorted_deduplicated";
		const std::string result_filename = "forward_" + zerofill_itos(N - 0) + "_boardlist";

		firststep_enumerate_legalmoves(N, reading_filename, writing_filename);

		renbansort(N, writing_filename, result_filename);

		std::error_code ec;
		if(std::filesystem::exists(writing_filename, ec)){
			std::filesystem::remove(writing_filename);
		}
		return 0;
	}
	else if (5 <= N && N <= 36 && (S == "c")) {
		const std::string result_filename = "forward_" + zerofill_itos(N - 0) + "_boardlist";
		uint64_t prev_code = 0;
		for(int64_t i = 0;; ++i){
			const std::string filename = result_filename + "_" + std::to_string(i) + ".txt";

			std::error_code ec;
			if (!std::filesystem::exists(filename, ec)){
				break;
			}
			const int64_t filesize = std::filesystem::file_size(filename);
			assert(filesize % 10 == 0);
			const int64_t linesize = filesize / 10;
			if(filesize == 0){
				std::cout << "warning: c file filesize==0. filename=" << filename << std::endl;
				return 0;
			}

			std::vector<char> tmp_string(filesize + 1);
			std::ifstream reading_file;
			reading_file.open(filename, std::ios::in);
			if (reading_file.fail()){
				std::cout << "c reading_file open failed. filename=" << filename << std::endl;
				assert(false);
			}
			reading_file.read(tmp_string.data(), filesize);
			reading_file.close();

			std::vector<uint64_t> codes(linesize);
#pragma omp parallel for schedule(static)
			for(int64_t j = 0; j < linesize; ++j){
				std::array<char, 9> line;
				for(int k = 0; k < 9; ++k){
					line[k] = tmp_string[j * 10 + k];
				}
				assert(tmp_string[j * 10 + 9] == '\n');
				codes[j] = encode_bb(decode_string_bb(line));
			}

			if(prev_code == codes[0]){
				std::cout << "duplicated code found. i="<< i <<", j= " << 0 << ", code=" << codes[0] << std::endl;
			}
			if(prev_code > codes[0]){
				std::cout << "code is not sorted. i="<< i <<", j= " << 0 << ", prev_code=" << prev_code << " code=" << codes[0] << std::endl;
			}

#pragma omp parallel for schedule(static)
			for(int64_t j = 1; j < linesize; ++j){
				if (codes[j - 1] >= codes[j]){
#pragma omp critical
					{
						if(codes[j - 1] == codes[j]){
							std::cout << "duplicated code found. i="<< i <<", j= " << j << ", code=" << codes[j] << ", codes[0]=" << codes[0] << std::endl;
						}
						if(codes[j - 1] > codes[j]){
							std::cout << "code is not sorted. i="<< i <<", j= " << j << ", prev_code=" << codes[j - 1] << " code=" << codes[j] << ", codes[0]=" << codes[0] << std::endl;
						}
					}
				}
			}
			prev_code = codes.back();
		}
		return 0;
	}
	else if(5 <= N && N <= 36 && (S == "s")) {
		const std::string reading_filename = "forward_" + zerofill_itos(N - 1) + "_boardlist";

		firststep_enumerate_legalmoves_countonly(N, reading_filename);
		return 0;
	}
	return 1;
}

int main(int argc, char *argv[]) {

	//test(42, 1000000);

	init_ternary_tables();

	if (argc != 3)return 1;
	const int N = std::stoi(argv[1]);
	const std::string S(argv[2]);

	return solve(N, S);



	/*
num_stone==5 : current_boards.size()==1
num_stone==6 : current_boards.size()==3
num_stone==7 : current_boards.size()==14
num_stone==8 : current_boards.size()==60
num_stone==9 : current_boards.size()==314
num_stone==10 : current_boards.size()==1632
num_stone==11 : current_boards.size()==9069
num_stone==12 : current_boards.size()==51964
num_stone==13 : current_boards.size()==292946
num_stone==14 : current_boards.size()==1706168
num_stone==15 : current_boards.size()==9289258
num_stone==16 : current_boards.size()==51072917
num_stone==17 : current_boards.size()==251070145
	*/


	return 0;
}
