
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
#include <cstdint>
#include <cstring>
#include<exception>
#include<functional>
#include<limits>
#include<queue>
#include<numeric>
#include<tuple>
#include<regex>
#include<random>
#include<filesystem>
#include <charconv>
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

void print_bb(const std::array<uint64_t, 2> bb){
	for(int i = 0; i < 64; ++i){
		if(bb[0] & (1ULL << i)){
			std::cout << "X";
		}else if(bb[1] & (1ULL << i)){
			std::cout << "O";
		}else{
			std::cout << "-";
		}
		if(i % 8 == 7){
			std::cout << std::endl;
		}
	}
}
std::string bb_to_obf67(const std::array<uint64_t, 2> bb) {
	std::string ans;
	assert((bb[0] & bb[1]) == 0);
	for (uint64_t i = 0; i < 64; ++i) {
		if (bb[0] & (1ULL << i)) {
			ans += "X";
		}
		else if (bb[1] & (1ULL << i)) {
			ans += "O";
		}
		else {
			ans += "-";
		}
	}
	return ans + " X;";
}
std::array<uint64_t, 2> obf67_to_bb(const std::string obf) {
	const std::regex obf67(R"([-OX]{64}\s[OX];)");
	assert(std::regex_match(obf, obf67));
	std::array<uint64_t, 2> bb{ 0,0 };
	const auto opponent = obf[65] == 'X' ? 'O' : 'X';
	for (uint64_t i = 0; i < 64; ++i) {
		if (obf[i] == obf[65]) {
			bb[0] |= 1ULL << i;
		}
		else if (obf[i] == opponent) {
			bb[1] |= 1ULL << i;
		}
		else assert(obf[i] == '-');
	}
	return bb;
}
std::array<uint64_t, 2> obf67_to_bb(const char* obf) {
	std::array<uint64_t, 2> bb{ 0,0 };
	const auto opponent = obf[65] == 'X' ? 'O' : 'X';
	for (uint64_t i = 0; i < 64; ++i) {
		if (obf[i] == obf[65]) {
			bb[0] |= 1ULL << i;
		}
		else if (obf[i] == opponent) {
			bb[1] |= 1ULL << i;
		}
		else assert(obf[i] == '-');
	}
	return bb;
}

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
	const __m256i tt3_ppoo = _mm256_or_si256(tt3lo_ppoo, tt3hi_ppoo);//この時点で、上位から順に(P横鏡映、P縦鏡映、O横鏡映、O縦鏡映)

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
	const __m256i rva5 = _mm256_add_epi32(rva4, _mm256_slli_epi64(rva3, 4));//この時点で、上位から順に(any、P逆転、any、O逆転)
	const __m256i rev_ppoo = _mm256_blend_epi32(rva5, bb0_ppoo, 0b11001100);//この時点で、上位から順に(Pそのまま、P逆転、Oそのまま、O逆転)

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
	const __m256i zz_t = _mm256_xor_si256(y2_t, _mm256_xor_si256(x3_t, _mm256_slli_epi64(x3_t, 28)));//この時点で、tt3_ppooの各要素を行列転置したもの
	const __m256i zz_r = _mm256_xor_si256(y2_r, _mm256_xor_si256(x3_r, _mm256_slli_epi64(x3_r, 28)));//この時点で、rev_ppooの各要素を行列転置したもの

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


uint64_t transpose(uint64_t b)
{
	uint64_t t;

	t = (b ^ (b >> 7)) & 0x00aa00aa00aa00aaULL;
	b = b ^ t ^ (t << 7);
	t = (b ^ (b >> 14)) & 0x0000cccc0000ccccULL;
	b = b ^ t ^ (t << 14);
	t = (b ^ (b >> 28)) & 0x00000000f0f0f0f0ULL;
	b = b ^ t ^ (t << 28);

	return b;
}
uint64_t vertical_mirror(uint64_t b)
{
	b = ((b >> 8) & 0x00FF00FF00FF00FFULL) | ((b << 8) & 0xFF00FF00FF00FF00ULL);
	b = ((b >> 16) & 0x0000FFFF0000FFFFULL) | ((b << 16) & 0xFFFF0000FFFF0000ULL);
	b = ((b >> 32) & 0x00000000FFFFFFFFULL) | ((b << 32) & 0xFFFFFFFF00000000ULL);
	return b;
}
uint64_t horizontal_mirror(uint64_t b)
{
	b = ((b >> 1) & 0x5555555555555555ULL) | ((b << 1) & 0xAAAAAAAAAAAAAAAAULL);
	b = ((b >> 2) & 0x3333333333333333ULL) | ((b << 2) & 0xCCCCCCCCCCCCCCCCULL);
	b = ((b >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((b << 4) & 0xF0F0F0F0F0F0F0F0ULL);

	return b;
}
int board_compare(const uint64_t player1, const uint64_t opponent1, const uint64_t player2, const uint64_t opponent2)
{
	if (player1 > player2) return 1;
	if (player1 < player2) return -1;
	if (opponent1 > opponent2) return 1;
	if (opponent1 < opponent2) return -1;
	return 0;
}
std::array<uint64_t, 2> board_symetry(uint64_t player, uint64_t opponent, const uint32_t code)
{
	if (code & 1) {
		player = horizontal_mirror(player);
		opponent = horizontal_mirror(opponent);
	}
	if (code & 2) {
		player = vertical_mirror(player);
		opponent = vertical_mirror(opponent);
	}
	if (code & 4) {
		player = transpose(player);
		opponent = transpose(opponent);
	}
	return std::array<uint64_t, 2>{player, opponent};
}

std::array<uint64_t, 2> board_unique_naive_with_code(const uint64_t player, const uint64_t opponent, uint32_t &symmetry_code)
{
	std::array<uint64_t, 2> unique{ player, opponent };
	symmetry_code = 0;


	for (int i = 1; i < 8; ++i) {
		const auto unique_candidate = board_symetry(player, opponent, i);
		if (board_compare(unique[0], unique[1], unique_candidate[0], unique_candidate[1]) > 0) {
			unique = unique_candidate;
			symmetry_code = i;
		}
	}

	return unique;
}

//std::array<uint64_t, 2> board_unique_naive_greycode(const uint64_t player, const uint64_t opponent) {
//	uint64_t p = player, o = opponent;
//	bool b;
//	const uint64_t code100p = transpose(player);
//	const uint64_t code100o = transpose(opponent);
//	b = (p < code100p) | ((p == code100p) & (o < code100o));
//	p = b ? p : code100p;
//	o = b ? o : code100o;
//	const uint64_t code001p = horizontal_mirror(player);
//	const uint64_t code001o = horizontal_mirror(opponent);
//	b = (p < code001p) | ((p == code001p) & (o < code001o));
//	p = b ? p : code001p;
//	o = b ? o : code001o;
//	const uint64_t code010p = vertical_mirror(player);
//	const uint64_t code010o = vertical_mirror(opponent);
//	b = (p < code010p) | ((p == code010p) & (o < code010o));
//	p = b ? p : code010p;
//	o = b ? o : code010o;
//	const uint64_t code101p = horizontal_mirror(code100p);
//	const uint64_t code101o = horizontal_mirror(code100o);
//	b = (p < code101p) | ((p == code101p) & (o < code101o));
//	p = b ? p : code101p;
//	o = b ? o : code101o;
//	const uint64_t code011p = horizontal_mirror(code010p);
//	const uint64_t code011o = horizontal_mirror(code010o);
//	b = (p < code011p) | ((p == code011p) & (o < code011o));
//	p = b ? p : code011p;
//	o = b ? o : code011o;
//	const uint64_t code110p = vertical_mirror(code100p);
//	const uint64_t code110o = vertical_mirror(code100o);
//	b = (p < code110p) | ((p == code110p) & (o < code110o));
//	p = b ? p : code110p;
//	o = b ? o : code110o;
//	const uint64_t code111p = vertical_mirror(code101p);
//	const uint64_t code111o = vertical_mirror(code101o);
//	b = (p < code111p) | ((p == code111p) & (o < code111o));
//	p = b ? p : code111p;
//	o = b ? o : code111o;
//	return {p, o};
//}

std::array<uint64_t, 2>board_unique(const uint64_t original_player, const uint64_t original_opponent) {
	std::array<uint64_t, 2> answer;
	board_unique(original_player, original_opponent, answer[0], answer[1]);
	return answer;
}

//uint64_t benchmark_unique()
//{
//	std::mt19937_64 rnd(12345);
//	std::array<uint64_t, 2> res{ rnd(), rnd() };
//	uint64_t result = 0;
//	for (int i = 0; i < 300'000'000; ++i) {
//		res = board_unique(result + res[0], result ^ res[1]);
//		result += res[0] + res[1];
//	}
//	return result;
//}
//uint64_t benchmark_unique_naive()
//{
//	std::mt19937_64 rnd(12345);
//	std::array<uint64_t, 2> res{ rnd(), rnd() };
//	uint64_t result = 0;
//	for (int i = 0; i < 300'000'000; ++i) {
//		res = board_unique_naive_greycode(result + res[0], result ^ res[1]);
//		result += res[0] + res[1];
//	}
//	return result;
//}

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
inline uint32_t bitscan_forward64(const uint64_t x, uint32_t *dest) {

	//xが非ゼロなら、立っているビットのうち最下位のものの位置をdestに代入して、非ゼロの値を返す。
	//xがゼロなら、ゼロを返す。このときのdestの値は未定義である。

#ifdef _MSC_VER
	return _BitScanForward64(reinterpret_cast<unsigned long *>(dest), x);
#else
	return x ? *dest = __builtin_ctzll(x), 1 : 0;
#endif

}
int32_t ComputeFinalScore(const uint64_t player, const uint64_t opponent) {

	//引数の盤面が即詰みだと仮定し、最終スコアを返す。

	const int32_t n_discs_p = _mm_popcnt_u64(player);
	const int32_t n_discs_o = _mm_popcnt_u64(opponent);

	int32_t score = n_discs_p - n_discs_o;

	//空白マスが残っている状態で両者とも打つ場所が無い場合は試合終了だが、
	//そのとき引き分けでないならば、空白マスは勝者のポイントに加算されるというルールがある。
	if (score < 0) score -= 36 - n_discs_p - n_discs_o;
	else if (score > 0) score += 36 - n_discs_p - n_discs_o;

	return score;
}



constexpr uint64_t MASK_EDGE8x8 = 0xFF81'8181'8181'81FFULL;
constexpr uint64_t MASK_8x8to6x6 = ~MASK_EDGE8x8;

const auto codebook = "56789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#_";//81文字
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


uint64_t wang_hash64(uint64_t key)
{
	//https://github.com/lh3/gwfa/blob/0dd07ff6a5d346294c8045b059b2ecbb1bfd879b/khash.h#L418
	key = ~key + (key << 21);
	key = key ^ key >> 24;
	key = (key + (key << 3)) + (key << 8);
	key = key ^ key >> 14;
	key = (key + (key << 2)) + (key << 4);
	key = key ^ key >> 28;
	key = key + (key << 31);
	return key;
}


namespace my_hash_function {
	struct hash_uint64_t {
		size_t operator()(const uint64_t &x) const {
			return wang_hash64(x);
		}
	};
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

std::array<uint64_t, 2>get_initial_position() {
	return std::array<uint64_t, 2>{ (1ULL << E4) | (1ULL << D5), (1ULL << D4) | (1ULL << E5) };
}


struct Bound {
	int8_t lowerbound;
	int8_t upperbound;

	Bound() {
		lowerbound = -36;
		upperbound = 36;
	}
	Bound(const int8_t new_lowerbound, const int8_t new_upperbound) {
		lowerbound = new_lowerbound;
		upperbound = new_upperbound;
		assert(-36 <= lowerbound && lowerbound <= upperbound && upperbound <= 36);
	}
	void update(const int8_t new_lowerbound, const int8_t new_upperbound) {
		if (lowerbound < new_lowerbound)lowerbound = new_lowerbound;
		if (new_upperbound < upperbound)upperbound = new_upperbound;
		assert(-36 <= lowerbound && lowerbound <= upperbound && upperbound <= 36);
	}
};

struct HashEntry {
	uint64_t hashed_code;
	uint64_t encoded_bb;
	Bound bound;

	HashEntry() {
		hashed_code = 0;
		bound.lowerbound = -36;
		bound.upperbound = 36;
	}
	HashEntry(const uint64_t _hashed_code, const uint64_t _encoded_bb) {
		hashed_code = _hashed_code;
		encoded_bb = _encoded_bb;
		bound.lowerbound = -36;
		bound.upperbound = 36;
	}
	HashEntry(const uint64_t _hashed_code, const uint64_t _encoded_bb, const int8_t _lowerbound, const int8_t _upperbound) {
		assert(-36 <= _lowerbound && _lowerbound <= _upperbound && _upperbound <= 36);
		hashed_code = _hashed_code;
		encoded_bb = _encoded_bb;
		bound.lowerbound = _lowerbound;
		bound.upperbound = _upperbound;
	}
};



class HashTable {

private:

	uint64_t hash_rank[8][256];

	void init_hash_rank() {
		std::mt19937_64 rnd(123456789);

		for (int i = 0; i < 8; ++i) for (int j = 0; j < 256; ++j) {
			for (;;) {
				hash_rank[i][j] = rnd();
				if (8 <= _mm_popcnt_u64(hash_rank[i][j]) && _mm_popcnt_u64(hash_rank[i][j]) <= 56)break;
			}
		}
	}

	uint64_t get_hash_code(uint64_t plaintext) {

		uint64_t h = hash_rank[0][plaintext % 256];

		for (int i = 1; i < 7; ++i) {
			plaintext /= 256;
			h ^= hash_rank[i][plaintext % 256];
		}

		h ^= hash_rank[7][plaintext / 256];

		return h;
	}

	static bool insert_(
		HashEntry &hash_entry, // Robin Hood Hashingのため、参照渡しでconstつけない。
		const int64_t table_bitlen_,
		std::vector<HashEntry>&hash_table_,
		std::vector<uint8_t>&distance_table_,
		std::vector<uint8_t>&signature_table_,
		int64_t &population_) {
		//hash_entryをテーブルに代入しようとする。代入できたらtrue, 失敗したらfalseを返す。
		//Robin Hood Hashingを行う。失敗した場合、引数として受け取ったhash_entryはテーブルに代入されていて、別のhash_entryを代入失敗するケースがありうる。
		//失敗した場合に代入できなかったentryがhash_entryに格納された状態でfalseを返す。

		uint64_t index = hash_entry.hashed_code & ((1ULL << table_bitlen_) - 1);
		uint8_t signature = uint8_t(hash_entry.hashed_code >> 57);

		for (uint64_t i = index; i < index + 32; ++i) {

			if (signature_table_[i] == 0x80) {
				//空の場所を見つけたら代入して終了

				hash_table_[i] = hash_entry;
				distance_table_[i] = i - index;
				signature_table_[i] = signature;
				++population_;
				return true;
			}
			else if (hash_table_[i].hashed_code == hash_entry.hashed_code) {
				//同じ局面が入っている場所を見つけたらアップデートして終了

				assert(signature_table_[i] == signature);

				hash_table_[i].bound.update(hash_entry.bound.lowerbound, hash_entry.bound.upperbound);

				return true;
			}
			else if (i - index > distance_table_[i]) {
				//今入れようとしている局面をpとする。ハッシュテーブル[i]には別の局面qの値が入っている。
				//pの距離がqの距離よりも遠いなら、qを取り出してから[i]にpを代入し、qを別の場所に入れる。(Robin Hood Hashing)

				HashEntry new_entry = hash_table_[i];
				uint64_t new_index = i - distance_table_[i];
				uint8_t new_signature = signature_table_[i];
				hash_table_[i] = hash_entry;
				distance_table_[i] = i - index;
				signature_table_[i] = signature;
				hash_entry = new_entry;
				index = new_index;
				signature = new_signature;
			}
		}

		return false;
	}

	std::vector<uint8_t>distance_table;
	int64_t table_bitlen; // hash_tableの現在の容量は2^(table_bitlen)
	int64_t population;

public:
	std::vector<HashEntry>hash_table;
	std::vector<uint8_t>signature_table;

	HashTable() {
		table_bitlen = 10;
		population = 0;
		hash_table.resize(((1ULL << table_bitlen) + 31));
		distance_table.resize((1ULL << table_bitlen) + 31, 0);
		signature_table.resize((1ULL << table_bitlen) + 31, 0x80);
		init_hash_rank();
	}
	HashTable(const int64_t _table_bitlen) {
		assert(10 <= _table_bitlen && _table_bitlen <= 50);
		table_bitlen = _table_bitlen;
		population = 0;
		hash_table.resize(((1ULL << table_bitlen) + 31));
		distance_table.resize((1ULL << table_bitlen) + 31, 0);
		signature_table.resize((1ULL << table_bitlen) + 31, 0x80);
		init_hash_rank();
	}

private:

	void extend() {

		for (int64_t new_table_bitlen = table_bitlen + 1;; ++new_table_bitlen) {

			std::vector<HashEntry>new_hashtable(((1ULL << new_table_bitlen) + 31));
			std::vector<uint8_t>new_distance_table((1ULL << new_table_bitlen) + 31, 0);
			std::vector<uint8_t>new_signature_table((1ULL << new_table_bitlen) + 31, 0x80);

			int64_t new_population = 0;

			bool flag = true;

			for (int i = 0; i < ((1ULL << table_bitlen) + 31); ++i) {
				if (signature_table[i] != 0x80) {
					HashEntry e = hash_table[i];
					if (!insert_(e, new_table_bitlen, new_hashtable, new_distance_table, new_signature_table, new_population)) {
						flag = false;
						break;
					}
				}
			}

			if (flag) {
				assert(new_population == population);
				table_bitlen = new_table_bitlen;
				hash_table.swap(new_hashtable);
				distance_table.swap(new_distance_table);
				signature_table.swap(new_signature_table);
				return;
			}
		}
		assert(false);
		return;
	}

	void insert_top(const uint64_t hashed_bb_code, const uint64_t encoded_bb, const int32_t lowerbound, const int32_t upperbound) {

		HashEntry entry(hashed_bb_code, encoded_bb, lowerbound, upperbound);

		if (!HashTable::insert_(entry, table_bitlen, hash_table, distance_table, signature_table, population)) {
			extend();
			insert_top(entry.hashed_code, entry.encoded_bb, entry.bound.lowerbound, entry.bound.upperbound);
		}
	}

public:

	int64_t size() {
		return population;
	}
	int64_t get_bitlen() {
		return table_bitlen;
	}

	void insert(const uint64_t bb_code, const int32_t lowerbound, const int32_t upperbound) {
		insert_top(get_hash_code(bb_code), bb_code, lowerbound, upperbound);
	}

	bool find(const uint64_t bb_code, Bound &bound) {

		//ハッシュテーブルに引数局面の情報があるか調べて、あればその情報を格納してtrueを返し、なければfalseを返す。

		//ナイーブな処理手順では、[index]から順番になめていって所望の局面かどうかを調べていき、所望の局面を得るか空白エントリに当たったら終了する。
		//でも今回はシグネチャ配列があるので効率的に計算できる。空白エントリはシグネチャが0x80であることを考慮しつつ、以下のようなAVX2のコードが書ける。

		const uint64_t hashcode = get_hash_code(bb_code);// wang_hash64(bb_code);
		const uint64_t index = hashcode & ((1ULL << table_bitlen) - 1);

		__m256i query_signature = _mm256_set1_epi8(int8_t(hashcode >> 57));
		__m256i table_signature = _mm256_loadu_si256((__m256i*)&signature_table.data()[index]);

		//[index + i]の情報が ↑のsignature_table.i8[i]に格納されているとして、↓のi桁目のbitに移されるとする。

		const uint64_t is_empty = uint32_t(_mm256_movemask_epi8(table_signature));
		const uint64_t is_positive = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(query_signature, table_signature)));

		uint64_t to_look = (is_empty ^ (is_empty - 1)) & is_positive;

		//[index+i]がシグネチャ陽性かどうかがis_positiveの下からi番目のビットにあるとする。
		//最初に当たる空白エントリより手前にあるシグネチャ陽性な局面の位置のビットボードが計算できる。to_lookがそれである。

		for (uint32_t i = 0; bitscan_forward64(to_look, &i); to_look &= to_look - 1) {
			const uint64_t pos = index + i;
			if (hash_table[pos].hashed_code != hashcode)continue;
			bound = hash_table[pos].bound;
			return true;
		}
		return false;
	}
};

class HashTable_unordered_map {

private:

	std::unordered_map<uint64_t, Bound, my_hash_function::hash_uint64_t>table;

public:

	HashTable_unordered_map() {}

	HashTable_unordered_map(const int64_t _table_bitlen) {}

	int64_t size() {
		return table.size();
	}
	int64_t get_bitlen() {
		return 0;
	}

	void insert(const uint64_t bb_code, const int32_t lowerbound, const int32_t upperbound) {
		Bound b(lowerbound, upperbound);
		table[bb_code] = b;
	}

	bool find(const uint64_t bb_code, Bound &bound) {

		//ハッシュテーブルに引数局面の情報があるか調べて、あればその情報を格納してtrueを返し、なければfalseを返す。

		if (table.find(bb_code) != table.end()) {
			bound = table[bb_code];
			return true;
		}
		return false;
	}

};




uint64_t get_full_lines(const uint64_t line, const int dir)
{

	// kogge-stone algorithm
	// 5 << + 5 >> + 7 & + 10 |
	// + better instruction independency
	uint64_t full_l, full_r, edge_l, edge_r;
	const uint64_t edge = 0xff818181818181ff;
	const int dir2 = dir << 1;
	const int dir4 = dir << 2;

	full_l = line & (edge | (line >> dir)); full_r = line & (edge | (line << dir));
	edge_l = edge | (edge >> dir);        edge_r = edge | (edge << dir);
	full_l &= edge_l | (full_l >> dir2);  full_r &= edge_r | (full_r << dir2);
	edge_l |= edge_l >> dir2;             edge_r |= edge_r << dir2;
	full_l &= edge_l | (full_l >> dir4);  full_r &= edge_r | (full_r << dir4);

	return full_r & full_l;
}
uint64_t get_full_lines_h(uint64_t full)
{
	full &= full >> 1;
	full &= full >> 2;
	full &= full >> 4;
	return (full & 0x0101010101010101ULL) * 0xFF;
}
uint64_t get_full_lines_v(uint64_t full)
{
	full &= (full >> 8) | (full << 56);	// ror 8
	full &= (full >> 16) | (full << 48);	// ror 16
	full &= (full >> 32) | (full << 32);	// ror 32
	return full;
}
int get_stability(const uint64_t P, const uint64_t O)
{
	uint64_t P_central, disc, full_h, full_v, full_d7, full_d9;
	uint64_t stable_h, stable_v, stable_d7, stable_d9, stable, old_stable;


	disc = (P | O | MASK_EDGE8x8);
	P_central = P;// (P & 0x007e7e7e7e7e7e00);

	full_h = get_full_lines_h(disc);
	full_v = get_full_lines_v(disc);
	full_d7 = get_full_lines(disc, 7);
	full_d9 = get_full_lines(disc, 9);

	// compute the exact stable edges (from precomputed tables)
	stable = MASK_EDGE8x8;//get_stable_edge(P, O);

	// add full lines
	stable |= (full_h & full_v & full_d7 & full_d9 & P_central);

	if (stable == 0)
		return 0;

	// now compute the other stable discs (ie discs touching another stable disc in each flipping direction).
	do {
		old_stable = stable;
		stable_h = ((stable >> 1) | (stable << 1) | full_h);
		stable_v = ((stable >> 8) | (stable << 8) | full_v);
		stable_d7 = ((stable >> 7) | (stable << 7) | full_d7);
		stable_d9 = ((stable >> 9) | (stable << 9) | full_d9);
		stable |= (stable_h & stable_v & stable_d7 & stable_d9 & P_central);
	} while (stable != old_stable);

	return _mm_popcnt_u64(stable) - 28; // 28 == _mm_popcnt_u64(MASK_EDGE8x8)
}

uint64_t get_some_potential_moves(const uint64_t O, const int dir)
{
	return (O << dir | O >> dir);
}
uint64_t get_potential_moves(const uint64_t P, const uint64_t O)
{
	return (get_some_potential_moves(O & 0x7E7E7E7E7E7E7E7EULL, 1) // horizontal
		| get_some_potential_moves(O & 0x00FFFFFFFFFFFF00ULL, 8)   // vertical
		| get_some_potential_moves(O & 0x007E7E7E7E7E7E00ULL, 7)   // diagonals
		| get_some_potential_moves(O & 0x007E7E7E7E7E7E00ULL, 9))
		& ~(P | O | MASK_EDGE8x8); // mask with empties
}
int bit_weighted_count(uint64_t v)
{
	constexpr uint64_t corner6x6 = (1ULL << B2) | (1ULL << G2) | (1ULL << B7) | (1ULL << G7);
	return _mm_popcnt_u64(v) + _mm_popcnt_u64(v & corner6x6);
}
const uint8_t SQUARE_VALUE[] = {
	 0,  0,  0,  0,  0,  0,  0,  0,
	 0, 18,  4, 12, 12,  4, 18,  0,
	 0,  4,  2,  8,  8,  2,  4,  0,
	 0, 12,  8,  0,  0,  8, 12,  0,
	 0, 12,  8,  0,  0,  8, 12,  0,
	 0,  4,  2,  8,  8,  2,  4,  0,
	 0, 18,  4, 12, 12,  4, 18,  0,
	 0,  0,  0,  0,  0,  0,  0,  0
};



//https://www.maths.nottingham.ac.uk/plp/pmzjff/Othello/6x6sol.html
//6x6オセロの最善進行が書かれている。d3始まりで書かれているので、f5始まりに変換する必要がある：
//d3c5d6e3f5f4e2d2c2e6e7g5c4c3g4g3f3c7b5d7b7b3c6b6f7f6b4b2g6g7paf2g2
//↓変換（＋pass除去）
//f5d6c5f4d3e3g4g5g6c4b4d2e6f6e2f2f3b6d7b5b7f7c6c7b3c3e7g7c2b2g3g2
//F5,D6,C5,F4,D3,E3,G4,G5,G6,C4,B4,D2,E6,F6,E2,F2,F3,B6,D7,B5,B7,F7,C6,C7,B3,C3,E7,G7,C2,B2,G3,G2
//37, 43, 34, 29, 19, 20, 30, 38, 46, 26, 25, 11, 44, 45, 12, 13, 21, 41, 51, 33, 49, 53, 42, 50, 17, 18, 52, 54, 10, 9, 22, 14

std::vector<int>solution_game_record{
	 F5,D6,C5,F4,D3,
	 E3,G4,G5,G6,C4,
	 B4,D2,E6,F6,E2,
	 F2,F3,B6,D7,B5,
	 B7,F7,C6,C7,B3,
	 C3,E7,G7,C2,B2,
	 G3,G2
};

std::array<std::array<uint64_t, 2>, 37>solution_unique_boards;
std::array<uint32_t, 37>unique_pos_record;
void init_solution_unique_boards() {
	std::array<uint64_t, 2> original_PO = get_initial_position();
	solution_unique_boards[4] = board_unique(original_PO[0], original_PO[1]);
	uint32_t symmetry_code = 0;
	std::array<uint64_t, 2> unique_PO = board_unique_naive_with_code(original_PO[0], original_PO[1], symmetry_code);
	assert(unique_PO[0] == solution_unique_boards[4][0] && unique_PO[1] == solution_unique_boards[4][1]);

	for (int i = 4; i < 36; ++i) {
		const int pos = solution_game_record[i - 4];
		unique_PO = board_unique_naive_with_code(original_PO[0], original_PO[1], symmetry_code);
		const std::array<uint64_t, 2> testingPO = board_unique(original_PO[0], original_PO[1]);
		assert(testingPO[0] == unique_PO[0] && testingPO[1] == unique_PO[1]);
		const uint64_t unique_pos_bb = board_symetry((1ULL << pos), 0, symmetry_code)[0];
		unique_pos_record[i] = _mm_popcnt_u64(unique_pos_bb - 1ULL);

		const uint64_t flipped = flip(original_PO[0], original_PO[1], pos);
		if (flipped == 0) {//pass
			assert(i == 34);
			std::swap(original_PO[0], original_PO[1]);
			--i;
			continue;
		}
		assert(flipped);
		original_PO[0] ^= (1ULL << pos) | flipped;
		original_PO[1] ^= flipped;
		solution_unique_boards[i + 1] = board_unique(original_PO[1], original_PO[0]);
		std::swap(original_PO[0], original_PO[1]);
	}
}

//HashTable_unordered_map transposition_table;
HashTable transposition_table;

int TT_THRESHOLD = 27;
int OPTIMAL_TT_THRESHOLD = 26;
int ENDGAME_THRESHOLD = 6;
constexpr int ENDGAME_THRESHOLD_UB = 6;

int move_scoring(const uint64_t P, const uint64_t O, const int pos) {
	assert(0 <= pos && pos < 64);

	const uint64_t flipped = flip(P, O, pos);
	assert(flipped);
	if (flipped == O)return 1 << 30; // wipeout
	const uint64_t next_player = O ^ flipped;
	const uint64_t next_opponent = P ^ (flipped | (1ULL << pos));

	int score = SQUARE_VALUE[pos];
	score += (36 - bit_weighted_count(get_potential_moves(next_player, next_opponent))) * (1 << 5);
	score += get_stability(next_opponent, next_player) * (1 << 11);
	score += (36 - bit_weighted_count(get_moves(next_player, next_opponent))) * (1 << 15);
	return score;
}

int32_t enhanced_transposition_cutoff(const uint64_t player, const uint64_t opponent, const int32_t beta, int64_t bb_moves, uint8_t &chosen_move) {
	assert(bb_moves);
	for (uint32_t index; bitscan_forward64(bb_moves, &index); bb_moves &= bb_moves - 1) {
		const uint64_t flipped = flip(player, opponent, index);
		assert(flipped);
		if(flipped == opponent){
			chosen_move = index;
			return 36;
		}
		const uint64_t next_player = opponent ^ flipped;
		const uint64_t next_opponent = player ^ (flipped | (1ULL << index));
		const uint64_t next_unique_code = encode_bb(board_unique(next_player, next_opponent));
		Bound b;
		if (transposition_table.find(next_unique_code, b)) {
			if (beta <= -b.upperbound) {
				chosen_move = index;
				return -b.upperbound;
			}
		}
	}
	return -36;
}

int get_stability_threhsold(const int n_empties, const bool is_nws) {
	return n_empties <= 3 ? 99 : ((is_nws ? -1 : -5) + n_empties);
}

//https://github.com/Nyanyan/Egaroucid/blob/641d1db88be43f4c1a76afa9532f6bf1a3102768/src/engine/common.hpp#L152
//constexpr uint64_t bit_around[64] = {
//	0x0000000000000302ULL, 0x0000000000000604ULL, 0x0000000000000e0aULL, 0x0000000000001c14ULL, 0x0000000000003828ULL, 0x0000000000007050ULL, 0x0000000000006020ULL, 0x000000000000c040ULL,
//	0x0000000000030200ULL, 0x0000000000060400ULL, 0x00000000000e0a00ULL, 0x00000000001c1400ULL, 0x0000000000382800ULL, 0x0000000000705000ULL, 0x0000000000602000ULL, 0x0000000000c04000ULL,
//	0x0000000003020300ULL, 0x0000000006040600ULL, 0x000000000e0a0e00ULL, 0x000000001c141c00ULL, 0x0000000038283800ULL, 0x0000000070507000ULL, 0x0000000060206000ULL, 0x00000000c040c000ULL,
//	0x0000000302030000ULL, 0x0000000604060000ULL, 0x0000000e0a0e0000ULL, 0x0000001c141c0000ULL, 0x0000003828380000ULL, 0x0000007050700000ULL, 0x0000006020600000ULL, 0x000000c040c00000ULL,
//	0x0000030203000000ULL, 0x0000060406000000ULL, 0x00000e0a0e000000ULL, 0x00001c141c000000ULL, 0x0000382838000000ULL, 0x0000705070000000ULL, 0x0000602060000000ULL, 0x0000c040c0000000ULL,
//	0x0003020300000000ULL, 0x0006040600000000ULL, 0x000e0a0e00000000ULL, 0x001c141c00000000ULL, 0x0038283800000000ULL, 0x0070507000000000ULL, 0x0060206000000000ULL, 0x00c040c000000000ULL,
//	0x0002030000000000ULL, 0x0004060000000000ULL, 0x000a0e0000000000ULL, 0x00141c0000000000ULL, 0x0028380000000000ULL, 0x0050700000000000ULL, 0x0020600000000000ULL, 0x0040c00000000000ULL,
//	0x0203000000000000ULL, 0x0406000000000000ULL, 0x0a0e000000000000ULL, 0x141c000000000000ULL, 0x2838000000000000ULL, 0x5070000000000000ULL, 0x2060000000000000ULL, 0x40c0000000000000ULL
//};
constexpr uint64_t bit_around[64] = {
	0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL,
	0x0000'0000'0000'0000ULL, 0x0000'0000'0006'0400ULL, 0x0000'0000'000c'0800ULL, 0x0000'0000'001c'1400ULL, 0x0000'0000'0038'2800ULL, 0x0000'0000'0030'1000ULL, 0x0000'0000'0060'2000ULL, 0x0000'0000'0000'0000ULL,
	0x0000'0000'0000'0000ULL, 0x0000'0000'0604'0000ULL, 0x0000'0000'0c08'0000ULL, 0x0000'0000'1c14'0000ULL, 0x0000'0000'3828'0000ULL, 0x0000'0000'3010'0000ULL, 0x0000'0000'6020'0000ULL, 0x0000'0000'0000'0000ULL,
	0x0000'0000'0000'0000ULL, 0x0000'0006'0406'0000ULL, 0x0000'000c'080c'0000ULL, 0x0000'001c'141c'0000ULL, 0x0000'0038'2838'0000ULL, 0x0000'0030'1030'0000ULL, 0x0000'0060'2060'0000ULL, 0x0000'0000'0000'0000ULL,
	0x0000'0000'0000'0000ULL, 0x0000'0604'0600'0000ULL, 0x0000'0c08'0c00'0000ULL, 0x0000'1c14'1c00'0000ULL, 0x0000'3828'3800'0000ULL, 0x0000'3010'3000'0000ULL, 0x0000'6020'6000'0000ULL, 0x0000'0000'0000'0000ULL,
	0x0000'0000'0000'0000ULL, 0x0000'0406'0000'0000ULL, 0x0000'080c'0000'0000ULL, 0x0000'141c'0000'0000ULL, 0x0000'2838'0000'0000ULL, 0x0000'1030'0000'0000ULL, 0x0000'2060'0000'0000ULL, 0x0000'0000'0000'0000ULL,
	0x0000'0000'0000'0000ULL, 0x0004'0600'0000'0000ULL, 0x0008'0c00'0000'0000ULL, 0x0014'1c00'0000'0000ULL, 0x0028'3800'0000'0000ULL, 0x0010'3000'0000'0000ULL, 0x0020'6000'0000'0000ULL, 0x0000'0000'0000'0000ULL,
	0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL, 0x0000'0000'0000'0000ULL
};

int32_t endgame_negascout(const uint64_t player, const uint64_t opponent, const int32_t alpha, const int32_t beta, std::array<int8_t, ENDGAME_THRESHOLD_UB> squares) {

	const int n_disc = _mm_popcnt_u64(player | opponent);
	if (n_disc == 36) {
		return ComputeFinalScore(player, opponent);
	}
	const int n_empty = 36 - n_disc;

	const auto stability_cutoff_threshold = get_stability_threhsold(36 - n_disc, alpha + 1 == beta);
	if (beta >= stability_cutoff_threshold) {
		const int score_upperbound = 36 - 2 * get_stability(opponent, player);
		if (score_upperbound <= alpha)return score_upperbound;
	}

	int32_t bestscore = -36, now_alpha = alpha, num_legalmove = 0;
	for (int num_searched = 0; num_searched < ENDGAME_THRESHOLD; ++num_searched) {
		if (squares[num_searched] < 0)continue;
		if ((bit_around[squares[num_searched]] & opponent) == 0)continue;

		const auto pos = squares[num_searched];

		const uint64_t flipped = flip(player, opponent, pos);
		if (flipped == 0)continue;
		if (flipped == opponent)return 36; // wipeout
		++num_legalmove;
		squares[num_searched] = -1;
		const uint64_t next_player = opponent ^ flipped;
		const uint64_t next_opponent = player ^ (flipped | (1ULL << pos));

		if (num_searched == 0) {
			bestscore = -endgame_negascout(next_player, next_opponent, -beta, -now_alpha, squares);
			if (beta <= bestscore) {
				squares[num_searched] = pos;
				return bestscore;
			}
			if (now_alpha < bestscore)now_alpha = bestscore;
		}
		else {
			auto score = -endgame_negascout(next_player, next_opponent, -now_alpha - 1, -now_alpha, squares);
			if (beta <= score) {
				squares[num_searched] = pos;
				return score;
			}
			if (now_alpha < score) {
				now_alpha = score;
				score = -endgame_negascout(next_player, next_opponent, -beta, -now_alpha, squares);
				if (beta <= score) {
					squares[num_searched] = pos;
					return score;
				}
				if (now_alpha < score)now_alpha = score;
			}
			if (bestscore < score)bestscore = score;
		}
		squares[num_searched] = pos;
	}
	if (num_legalmove == 0) {
		if ((get_moves(opponent, player) & MASK_8x8to6x6) != 0) { // pass
			return -endgame_negascout(opponent, player, -beta, -alpha, squares);
		}
		else { // game over
			return ComputeFinalScore(player, opponent);
		}
	}

	return bestscore;
}

int32_t entry_endgame_negascout(const uint64_t player, const uint64_t opponent, const int32_t alpha, const int32_t beta) {

	const int n_empty = 36 - _mm_popcnt_u64(player | opponent);
	uint64_t bb_empty = ~(player | opponent | MASK_EDGE8x8);

	std::array<int8_t, ENDGAME_THRESHOLD_UB>squares;
	int i = 0;
	for (uint32_t index; bitscan_forward64(bb_empty, &index); bb_empty &= bb_empty - 1) {
		squares[i++] = uint8_t(index);
	}
	for (; i < ENDGAME_THRESHOLD; ++i)squares[i] = -1;

	return endgame_negascout(player, opponent, alpha, beta, squares);
}

enum {
	NEGASCOUT_PV,
	NEGASCOUT_CHOOSE_GOOD,
	//非自明な局面かつendgame以前であれば、
	//  引数chosen_moveにはbeta cutされうるならされる手を必ず格納するが、最善手であることは保証しない。
	//  cutされないなら必ず最善手を格納する。
	//    (alpha==-36 && beta==36 で呼べば必ず最善手が返されることに注意）

	NEGASCOUT_MISC//chosen_moveになにを格納するか保証しない。
};
int32_t negascout(const uint64_t _player, const uint64_t _opponent, const int32_t alpha, const int32_t beta, const int32_t node_kind, uint8_t &chosen_move) {
	const std::array<uint64_t, 2> unique_board = board_unique(_player, _opponent);
	const uint64_t player = unique_board[0];
	const uint64_t opponent = unique_board[1];
	chosen_move = 0xFF;

	const auto n_disc = _mm_popcnt_u64(player | opponent);
	if (n_disc == 36) {
		return ComputeFinalScore(player, opponent);
	}

	const auto n_empty = 36 - n_disc;
	if (n_empty <= ENDGAME_THRESHOLD)return entry_endgame_negascout(player, opponent, alpha, beta);

	Bound b;
	const uint64_t unique_code = encode_bb(unique_board);
	if (node_kind == NEGASCOUT_MISC) {
		if(n_disc < TT_THRESHOLD){
			// transposition tableを確認して、答えがあればそれを返す
			if (transposition_table.find(unique_code, b)) {
				if (b.lowerbound == b.upperbound) return b.upperbound;
				if (b.upperbound <= alpha)return b.upperbound;
				if (beta <= b.lowerbound)return b.lowerbound;
			}
		}
		//stability cutoff
		const auto stability_cutoff_threshold = get_stability_threhsold(36 - n_disc, alpha + 1 == beta);
		if (beta >= stability_cutoff_threshold) {
			const int score_upperbound = 36 - 2 * get_stability(opponent, player);
			if (score_upperbound <= alpha)return score_upperbound;
		}
	}

	uint64_t bb_moves = get_moves(player, opponent) & MASK_8x8to6x6;

	if (bb_moves == 0) {
		uint64_t bb_next = get_moves(opponent, player) & MASK_8x8to6x6;
		if (bb_next != 0) { // pass

			//passの直後にwipeoutされるかどうか調べる。
			for (uint32_t index; bitscan_forward64(bb_next, &index); bb_next &= bb_next - 1) {
				const uint64_t flipped = flip(opponent, player, index);
				assert(flipped);
				if (flipped == player) { // wipeout
					if (n_disc < TT_THRESHOLD) {
						transposition_table.insert(unique_code, -36, -36);
						transposition_table.insert(encode_bb(board_unique(_opponent, _player)), 36, 36);
					}
					chosen_move = index;
					return -36;
				}
			}
			const int score = -negascout(opponent, player, -beta, -alpha, node_kind, chosen_move);
			if (n_disc < TT_THRESHOLD) {
				if (alpha < score && score < beta) {
					b.update(score, score);
				}
				else if (score <= alpha) {
					b.update(-36, score);
				}
				else if (beta <= score) {
					b.update(score, 36);
				}
				else assert(false);
				transposition_table.insert(unique_code, b.lowerbound, b.upperbound);
			}
			return score;
		}
		else { // game over (自明な局面だった)
			const int score = ComputeFinalScore(player, opponent);
			if (n_disc < TT_THRESHOLD) {
				transposition_table.insert(unique_code, score, score);
			}
			return score;
		}
	}
	else{
		//wipeoutする手が存在するか調べる。
		uint64_t bb_ = bb_moves;
		for (uint32_t index; bitscan_forward64(bb_, &index); bb_ &= bb_ - 1) {
			const uint64_t flipped = flip(player, opponent, index);
			assert(flipped);
			if (flipped == opponent) { // wipeout
				if (n_disc < TT_THRESHOLD) {
					transposition_table.insert(unique_code, 36, 36);
				}
				chosen_move = index;
				return 36;
			}
		}
	}

	if (node_kind == NEGASCOUT_MISC || node_kind == NEGASCOUT_CHOOSE_GOOD) {
		if (n_disc + 1 < TT_THRESHOLD) {
			const auto etc_score = enhanced_transposition_cutoff(player, opponent, beta, bb_moves, chosen_move);
			if (beta <= etc_score)return etc_score;
		}
	}

	if (node_kind == NEGASCOUT_PV) {
		assert(bb_moves & (1ULL << unique_pos_record[n_disc]));
	}

	std::array<uint8_t, 36>moves;
	std::array<int64_t, 36>move_score;
	for (int i = 0; i < 36; ++i) {
		moves[i] = 0;
		move_score[i] = 0;
	}
	uint64_t movenum = 0;

	for (uint32_t index; bitscan_forward64(bb_moves, &index); bb_moves &= bb_moves - 1) {
		moves[movenum] = uint8_t(index);
		if (node_kind == NEGASCOUT_PV && index == unique_pos_record[n_disc]) {
			move_score[movenum] = 1LL << 50;
		}
		else move_score[movenum] = move_scoring(player, opponent, index);
		++movenum;
	}

	int32_t bestscore = -36, now_alpha = alpha;
	uint8_t chosen_move_now = 0xFF;
	for (int num_searched = 0; movenum > 0; ++num_searched) {

		int64_t maxscore = move_score[0];
		int maxindex = 0;
		for (int i = 1; i < movenum; ++i)if (move_score[i] > maxscore) {
			maxscore = move_score[i];
			maxindex = i;
		}

		const auto pos = moves[maxindex];
		moves[maxindex] = moves[--movenum];

		const uint64_t flipped = flip(player, opponent, pos);
		assert(flipped);
		assert(flipped != opponent); // wipeoutできないことが事前にチェック済みであることを仮定する。
		const uint64_t next_player = opponent ^ flipped;
		const uint64_t next_opponent = player ^ (flipped | (1ULL << pos));

		if (num_searched == 0) {
			const int32_t next_node_kind = (node_kind == NEGASCOUT_PV) ? NEGASCOUT_PV : NEGASCOUT_MISC;
			bestscore = -negascout(next_player, next_opponent, -beta, -now_alpha, next_node_kind, chosen_move);
			if (beta <= bestscore) {
				if (n_disc < TT_THRESHOLD) {
					b.update(bestscore, 36);
					transposition_table.insert(unique_code, b.lowerbound, b.upperbound);
				}
				chosen_move = pos;
				return bestscore;
			}
			if (now_alpha < bestscore)now_alpha = bestscore;
			chosen_move_now = pos;
		}
		else {
			int32_t score = -negascout(next_player, next_opponent, -now_alpha - 1, -now_alpha, NEGASCOUT_MISC, chosen_move);
			if (beta <= score) {
				if (n_disc < TT_THRESHOLD) {
					b.update(score, 36);
					transposition_table.insert(unique_code, b.lowerbound, b.upperbound);
				}
				chosen_move = pos;
				return score;
			}
			if (now_alpha < score) {
				now_alpha = score;
				score = -negascout(next_player, next_opponent, -beta, -now_alpha, NEGASCOUT_MISC, chosen_move);
				if (beta <= score) {
					if (n_disc < TT_THRESHOLD) {
						b.update(score, 36);
						transposition_table.insert(unique_code, b.lowerbound, b.upperbound);
					}

					chosen_move = pos;
					return score;
				}
				if (now_alpha < score)now_alpha = score;
			}
			if (bestscore < score) {
				bestscore = score;
				chosen_move_now = pos;
			}
		}
	}

	if (n_disc < TT_THRESHOLD) {
		if (alpha < bestscore && bestscore < beta) {
			b.update(bestscore, bestscore);
		}
		else if (bestscore <= alpha) {
			b.update(-36, bestscore);
		}
		else assert(false);
		transposition_table.insert(unique_code, b.lowerbound, b.upperbound);
	}
	chosen_move = chosen_move_now;
	return bestscore;
}

enum {
	OPTIMAL_AB_PV_NODE = 1,
	OPTIMAL_AB_ALL_NODE = 2,
	OPTIMAL_AB_CUT_NODE = 4
};
int get_optimal_ab_child_node_kind(const int parent_kind, bool is_child_first) {
	switch (parent_kind) {
	case OPTIMAL_AB_PV_NODE:
		return is_child_first ? OPTIMAL_AB_PV_NODE : OPTIMAL_AB_CUT_NODE;
	case OPTIMAL_AB_ALL_NODE:
		return OPTIMAL_AB_CUT_NODE;
	case OPTIMAL_AB_CUT_NODE:
		return OPTIMAL_AB_ALL_NODE;
	default:
		assert(false);
	}
	assert(false);
	return 1;
}

std::unordered_map<uint64_t, std::pair<Bound, uint8_t>, my_hash_function::hash_uint64_t>optimal_ab_transposition_table;//second.secondはnode_kind
int32_t optimal_alphabeta(const uint64_t _player, const uint64_t _opponent, int32_t alpha, int32_t beta, const int node_kind) {
	const std::array<uint64_t, 2> unique_board = board_unique(_player, _opponent);
	const uint64_t player = unique_board[0];
	const uint64_t opponent = unique_board[1];

	const auto n_disc = _mm_popcnt_u64(player | opponent);
	if (n_disc == 36) {
		return ComputeFinalScore(player, opponent);
	}

	uint8_t chosen_move = 0xFF;
	Bound b;
	uint8_t tt_node_kind = 0;
	uint64_t unique_code = 0;
	if (n_disc < OPTIMAL_TT_THRESHOLD) {
		unique_code = encode_bb(unique_board);
		if (optimal_ab_transposition_table.find(unique_code) != optimal_ab_transposition_table.end()) {
			assert(node_kind != OPTIMAL_AB_PV_NODE); // PV nodeではないはず
			b = optimal_ab_transposition_table[unique_code].first;
			tt_node_kind = optimal_ab_transposition_table[unique_code].second;

			if (node_kind == OPTIMAL_AB_ALL_NODE || node_kind == OPTIMAL_AB_CUT_NODE) {
				if (b.lowerbound == b.upperbound) return b.upperbound;
				if (b.upperbound <= alpha)return b.upperbound;
				if (beta <= b.lowerbound)return b.lowerbound;
			}
			else assert(false);
		}
	}
	else {
		return negascout(player, opponent, alpha, beta, (node_kind == OPTIMAL_AB_PV_NODE) ? NEGASCOUT_PV : NEGASCOUT_MISC, chosen_move);
	}

	uint64_t bb_moves = get_moves(player, opponent) & MASK_8x8to6x6;

	if (bb_moves == 0) {
		uint64_t bb_next = get_moves(opponent, player) & MASK_8x8to6x6;
		if (bb_next != 0) { // pass
			return -optimal_alphabeta(opponent, player, -beta, -alpha, get_optimal_ab_child_node_kind(node_kind, true));
		}
		else { // game over (自明な局面だった)
			return ComputeFinalScore(player, opponent);
		}
	}


	std::array<uint8_t, 36>moves;
	std::array<int64_t, 36>move_score;
	for (int i = 0; i < 36; ++i) {
		moves[i] = 0;
		move_score[i] = 0;
	}
	uint64_t movenum = 0;

	uint64_t pv_move_code = 0;
	if (node_kind == OPTIMAL_AB_PV_NODE) {
		//6x6オセロの"本筋"の中にいるなら、最初に探索する手は確定される。
		assert(bb_moves & (1ULL << unique_pos_record[n_disc])); // optimality
		const uint64_t flipped = flip(player, opponent, unique_pos_record[n_disc]);
		assert(flipped);
		const uint64_t next_player = opponent ^ flipped;
		const uint64_t next_opponent = player ^ (flipped | (1ULL << unique_pos_record[n_disc]));
		pv_move_code = encode_bb({ next_player, next_opponent });
	}
	else if (node_kind == OPTIMAL_AB_CUT_NODE) {
		assert(TT_THRESHOLD > OPTIMAL_TT_THRESHOLD);
		negascout(player, opponent, beta - 1, beta, NEGASCOUT_CHOOSE_GOOD, chosen_move);

		//null-windowのnegascout探索をNEGASCOUT_CHOOSE_GOODでしているので、chosen_moveにはbeta_cutされる手が入っている。
		assert(chosen_move < 64);
		assert(bb_moves & (1ULL << chosen_move));
	}

	bool pv_found = false;

	for (uint32_t index; bitscan_forward64(bb_moves, &index); bb_moves &= bb_moves - 1) {
		//合法手のorderingのためのscoringをする。

		moves[movenum] = uint8_t(index);
		const uint64_t flipped = flip(player, opponent, index);
		assert(flipped);
		const uint64_t next_player = opponent ^ flipped;
		const uint64_t next_opponent = player ^ (flipped | (1ULL << index));

		if (node_kind == OPTIMAL_AB_PV_NODE) {
			//PVノードの場合、"本筋"を最初に探索する。それ以外の手の探索順序はどうであっても最適性は保たれる。
			if (pv_move_code == encode_bb({ next_player, next_opponent })) {
				assert(pv_found == false);
				move_score[movenum] = 1LL << 50;
				pv_found = true;
			}
			else move_score[movenum] = move_scoring(player, opponent, index);
		}
		else if (node_kind == OPTIMAL_AB_ALL_NODE) {
			//ALLノードの場合、どういう順に探索しようとも"最適性"が保たれる。
			move_score[movenum] = move_scoring(player, opponent, index);
		}
		else if (node_kind == OPTIMAL_AB_CUT_NODE) {
			//CUTノードの場合、fail-highする手を最初に探索すれば"最適性"が保たれる。
			//2手目以降はどういう順に探索しようとも"最適性"が保たれる。

			//上でnegascout探索をしているので、chosen_moveには"最適性"を保つための手が入っている。
			if (chosen_move == moves[movenum]) {
				move_score[movenum] = 1LL << 50;
				pv_found = true;
			}
			else move_score[movenum] = move_scoring(player, opponent, index);
		}
		else assert(false);
		++movenum;
	}

	if (node_kind == OPTIMAL_AB_PV_NODE || node_kind == OPTIMAL_AB_CUT_NODE)assert(pv_found);

	int32_t bestscore = -36, now_alpha = alpha;
	for (int num_searched = 0; movenum > 0; ++num_searched) {

		int64_t maxscore = move_score[0];
		int maxindex = 0;
		for (int i = 1; i < movenum; ++i)if (move_score[i] > maxscore) {
			maxscore = move_score[i];
			maxindex = i;
		}

		const auto pos = moves[maxindex];
		moves[maxindex] = moves[--movenum];

		const uint64_t flipped = flip(player, opponent, pos);
		assert(flipped);
		const uint64_t next_player = opponent ^ flipped;
		const uint64_t next_opponent = player ^ (flipped | (1ULL << pos));

		if (num_searched == 0) {
			const auto child_node_kind = get_optimal_ab_child_node_kind(node_kind, true);
			// if (child_node_kind == OPTIMAL_AB_PV_NODE) {
			// 	bestscore = -optimal_alphabeta(next_player, next_opponent, -36, 36, child_node_kind);
			// }
			// else {
				bestscore = -optimal_alphabeta(next_player, next_opponent, -beta, -now_alpha, child_node_kind);
			// }
			if (beta <= bestscore) {
				if (n_disc < OPTIMAL_TT_THRESHOLD) {
					b.update(bestscore, 36);
					optimal_ab_transposition_table[unique_code] = std::make_pair(b, tt_node_kind | uint8_t(node_kind));
				}
				return bestscore;
			}
			assert(node_kind != OPTIMAL_AB_CUT_NODE);//optimality
			if (now_alpha < bestscore)now_alpha = bestscore;
		}
		else {
			const auto child_node_kind = get_optimal_ab_child_node_kind(node_kind, false);
			assert(child_node_kind != OPTIMAL_AB_PV_NODE);//optimality
			int32_t score = 0;

			score = -optimal_alphabeta(next_player, next_opponent, -now_alpha - 1, -now_alpha, child_node_kind);

			if (beta <= score) {
				if (n_disc < OPTIMAL_TT_THRESHOLD) {
					b.update(score, 36);
					optimal_ab_transposition_table[unique_code] = std::make_pair(b, tt_node_kind | uint8_t(node_kind));
				}
				return score;
			}
			else if (now_alpha < score) {

				assert(false);//optimality

				// now_alpha = score;
				// score = -optimal_alphabeta(next_player, next_opponent, -beta, -now_alpha, get_optimal_ab_child_node_kind(node_kind, true));
				// if (beta <= score) {
				// 	if (n_disc < OPTIMAL_TT_THRESHOLD) {
				// 		b.update(score, 36);
				// 		optimal_ab_transposition_table[unique_code] = std::make_pair(b, tt_node_kind | uint8_t(node_kind));
				// 	}
				// 	return score;
				// }
				// if (now_alpha < score)now_alpha = score;
			}
			if (bestscore < score) {
				bestscore = score;
			}
		}
	}

	if (n_disc < OPTIMAL_TT_THRESHOLD) {
		if (node_kind == OPTIMAL_AB_PV_NODE) {
			assert(alpha == -36 && beta == 36);
		}
		if (alpha < bestscore && bestscore < beta) {
			b.update(bestscore, bestscore);
		}
		else if (bestscore <= alpha) {
			b.update(-36, bestscore);
		}
		else assert(false);
		optimal_ab_transposition_table[unique_code] = std::make_pair(b, tt_node_kind | uint8_t(node_kind));
	}

	return bestscore;


}


enum {
	REOPENING_AB_PV_NODE = 1,
	REOPENING_AB_ALL_DASH_NODE = 2,
	REOPENING_AB_PV_DASH_NODE = 4,
	REOPENING_AB_ALL_NODE = 8,
	REOPENING_AB_CUT_NODE = 16
};
int get_reopening_ab_child_node_kind(const int parent_kind, bool is_child_first) {
	switch (parent_kind) {
	case REOPENING_AB_PV_NODE:
		return is_child_first ? REOPENING_AB_PV_NODE : REOPENING_AB_ALL_DASH_NODE;
	case REOPENING_AB_ALL_DASH_NODE:
		return is_child_first ? REOPENING_AB_PV_DASH_NODE : REOPENING_AB_CUT_NODE;
	case REOPENING_AB_PV_DASH_NODE:
		return REOPENING_AB_ALL_DASH_NODE;
	case REOPENING_AB_ALL_NODE:
		return REOPENING_AB_CUT_NODE;
	case REOPENING_AB_CUT_NODE:
		return REOPENING_AB_ALL_NODE;
	default:
		assert(false);
	}
	assert(false);
	return 1;
}
std::unordered_map<uint64_t, std::pair<Bound, uint8_t>, my_hash_function::hash_uint64_t>optimal_reopening_ab_transposition_table;//second.secondはnode_kind
int32_t optimal_reopening_alphabeta(const uint64_t _player, const uint64_t _opponent, int32_t alpha, int32_t beta, const int node_kind) {
	const std::array<uint64_t, 2> unique_board = board_unique(_player, _opponent);
	const uint64_t player = unique_board[0];
	const uint64_t opponent = unique_board[1];

	const auto n_disc = _mm_popcnt_u64(player | opponent);
	if (n_disc == 36) {
		return ComputeFinalScore(player, opponent);
	}

	uint8_t chosen_move = 0xFF;
	Bound b;
	uint8_t tt_node_kind = 0;
	uint64_t unique_code = 0;
	if (n_disc < OPTIMAL_TT_THRESHOLD) {
		unique_code = encode_bb(unique_board);
		if (optimal_reopening_ab_transposition_table.find(unique_code) != optimal_reopening_ab_transposition_table.end()) {
			assert(node_kind != REOPENING_AB_PV_NODE); // PV nodeではないはず
			b = optimal_reopening_ab_transposition_table[unique_code].first;
			tt_node_kind = optimal_reopening_ab_transposition_table[unique_code].second;

			if (node_kind == REOPENING_AB_PV_DASH_NODE) {
				if (tt_node_kind & (REOPENING_AB_PV_NODE | REOPENING_AB_PV_DASH_NODE)) {
					assert(b.lowerbound == b.upperbound);
					return b.upperbound;
				}
			}
			else if (node_kind == REOPENING_AB_ALL_DASH_NODE) {
				if (tt_node_kind & (REOPENING_AB_PV_NODE | REOPENING_AB_ALL_DASH_NODE)) {
					assert(b.lowerbound == b.upperbound);
					return b.upperbound;
				}
			}
			else if (node_kind == REOPENING_AB_ALL_NODE || node_kind == REOPENING_AB_CUT_NODE) {
				if (b.lowerbound == b.upperbound) return b.upperbound;
				if (b.upperbound <= alpha)return b.upperbound;
				if (beta <= b.lowerbound)return b.lowerbound;
			}
			else assert(false);
		}
	}
	else {
		return negascout(player, opponent, alpha, beta, (node_kind == REOPENING_AB_PV_NODE) ? NEGASCOUT_PV : NEGASCOUT_MISC, chosen_move);
	}

	uint64_t bb_moves = get_moves(player, opponent) & MASK_8x8to6x6;

	if (bb_moves == 0) {
		uint64_t bb_next = get_moves(opponent, player) & MASK_8x8to6x6;
		if (bb_next != 0) { // pass
			return -optimal_reopening_alphabeta(opponent, player, -beta, -alpha, get_reopening_ab_child_node_kind(node_kind, true));
		}
		else { // game over (自明な局面だった)
			return ComputeFinalScore(player, opponent);
		}
	}


	std::array<uint8_t, 36>moves;
	std::array<int64_t, 36>move_score;
	for (int i = 0; i < 36; ++i) {
		moves[i] = 0;
		move_score[i] = 0;
	}
	uint64_t movenum = 0;

	uint64_t pv_move_code = 0;
	if (node_kind == REOPENING_AB_PV_NODE) {
		//6x6オセロの"本筋"の中にいるなら、最初に探索する手は確定される。
		assert(bb_moves & (1ULL << unique_pos_record[n_disc])); // optimality
		const uint64_t flipped = flip(player, opponent, unique_pos_record[n_disc]);
		assert(flipped);
		const uint64_t next_player = opponent ^ flipped;
		const uint64_t next_opponent = player ^ (flipped | (1ULL << unique_pos_record[n_disc]));
		pv_move_code = encode_bb({ next_player, next_opponent });
	}
	else if (node_kind == REOPENING_AB_ALL_DASH_NODE) {
		assert(TT_THRESHOLD > OPTIMAL_TT_THRESHOLD);
		negascout(player, opponent, -36, 36, NEGASCOUT_CHOOSE_GOOD, chosen_move);

		//full-windowのnegascout探索をNEGASCOUT_CHOOSE_GOODでしているので、chosen_moveに最善手が入っている。
		assert(chosen_move < 64);
		assert(bb_moves & (1ULL << chosen_move));
	}
	else if (node_kind == REOPENING_AB_CUT_NODE) {
		assert(TT_THRESHOLD > OPTIMAL_TT_THRESHOLD);
		negascout(player, opponent, beta - 1, beta, NEGASCOUT_CHOOSE_GOOD, chosen_move);

		//null-windowのnegascout探索をNEGASCOUT_CHOOSE_GOODでしているので、chosen_moveにはbeta_cutされる手が入っている。
		assert(chosen_move < 64);
		assert(bb_moves & (1ULL << chosen_move));
	}

	bool pv_found = false;

	for (uint32_t index; bitscan_forward64(bb_moves, &index); bb_moves &= bb_moves - 1) {
		//合法手のorderingのためのscoringをする。

		moves[movenum] = uint8_t(index);
		const uint64_t flipped = flip(player, opponent, index);
		assert(flipped);
		const uint64_t next_player = opponent ^ flipped;
		const uint64_t next_opponent = player ^ (flipped | (1ULL << index));

		if (node_kind == REOPENING_AB_PV_NODE) {
			//PVノードの場合、"本筋"を最初に探索する。それ以外の手の探索順序はどうであっても最適性は保たれる。
			if (pv_move_code == encode_bb({ next_player, next_opponent })) {
				assert(pv_found == false);
				move_score[movenum] = 1LL << 50;
				pv_found = true;
			}
			else move_score[movenum] = move_scoring(player, opponent, index);
		}
		else if (node_kind == REOPENING_AB_ALL_NODE || node_kind == REOPENING_AB_PV_DASH_NODE) {
			//ALLノードまたはP'ノードの場合、どういう順に探索しようとも"最適性"が保たれる。
			move_score[movenum] = move_scoring(player, opponent, index);
		}
		else if (node_kind == REOPENING_AB_CUT_NODE || node_kind == REOPENING_AB_ALL_DASH_NODE) {
			//CUTノードの場合、fail-highする手を最初に探索すれば"最適性"が保たれる。
			//A'ノードの場合、"最適性"を保つためには最善手を最初に探索する必要がある。
			//いずれも2手目以降はどういう順に探索しようとも"最適性"が保たれる。

			//上でnegascout探索をしているので、chosen_moveには"最適性"を保つための手が入っている。
			if (chosen_move == moves[movenum]) {
				move_score[movenum] = 1LL << 50;
				pv_found = true;
			}
			else move_score[movenum] = move_scoring(player, opponent, index);
		}
		else assert(false);
		++movenum;
	}

	if (node_kind == REOPENING_AB_PV_NODE || node_kind == REOPENING_AB_ALL_DASH_NODE || node_kind == REOPENING_AB_CUT_NODE)assert(pv_found);

	int32_t bestscore = -36, now_alpha = alpha;
	for (int num_searched = 0; movenum > 0; ++num_searched) {

		int64_t maxscore = move_score[0];
		int maxindex = 0;
		for (int i = 1; i < movenum; ++i)if (move_score[i] > maxscore) {
			maxscore = move_score[i];
			maxindex = i;
		}

		const auto pos = moves[maxindex];
		moves[maxindex] = moves[--movenum];

		const uint64_t flipped = flip(player, opponent, pos);
		assert(flipped);
		const uint64_t next_player = opponent ^ flipped;
		const uint64_t next_opponent = player ^ (flipped | (1ULL << pos));

		if (num_searched == 0) {
			const auto child_node_kind = get_reopening_ab_child_node_kind(node_kind, true);
			if (child_node_kind == REOPENING_AB_PV_NODE || child_node_kind == REOPENING_AB_ALL_DASH_NODE) {
				bestscore = -optimal_reopening_alphabeta(next_player, next_opponent, -36, 36, child_node_kind);
			}
			else {
				bestscore = -optimal_reopening_alphabeta(next_player, next_opponent, -beta, -now_alpha, child_node_kind);
			}
			if (beta <= bestscore) {
				if (n_disc < OPTIMAL_TT_THRESHOLD) {
					b.update(bestscore, 36);
					optimal_reopening_ab_transposition_table[unique_code] = std::make_pair(b, tt_node_kind | uint8_t(node_kind));
				}
				return bestscore;
			}
			assert(node_kind != REOPENING_AB_CUT_NODE);//optimality
			if (now_alpha < bestscore)now_alpha = bestscore;
		}
		else {
			const auto child_node_kind = get_reopening_ab_child_node_kind(node_kind, false);
			assert(child_node_kind != REOPENING_AB_PV_NODE);//optimality
			int32_t score = 0;
			if (child_node_kind == REOPENING_AB_ALL_DASH_NODE) {
				score = -optimal_reopening_alphabeta(next_player, next_opponent, -36, 36, child_node_kind);
			}
			else {
				score = -optimal_reopening_alphabeta(next_player, next_opponent, -now_alpha - 1, -now_alpha, child_node_kind);
			}
			if (beta <= score) {
				if (n_disc < OPTIMAL_TT_THRESHOLD) {
					b.update(score, 36);
					optimal_reopening_ab_transposition_table[unique_code] = std::make_pair(b, tt_node_kind | uint8_t(node_kind));
				}
				return score;
			}
			if (child_node_kind == REOPENING_AB_ALL_DASH_NODE) {
				if (now_alpha < score)now_alpha = score;
			}
			else if (now_alpha < score) {

				assert(false);//optimality

				// now_alpha = score;
				// score = -optimal_reopening_alphabeta(next_player, next_opponent, -beta, -now_alpha, get_reopening_ab_child_node_kind(node_kind, true));
				// if (beta <= score) {
				// 	if (n_disc < OPTIMAL_TT_THRESHOLD) {
				// 		b.update(score, 36);
				// 		optimal_reopening_ab_transposition_table[unique_code] = std::make_pair(b, tt_node_kind | uint8_t(node_kind));
				// 	}
				// 	return score;
				// }
				// if (now_alpha < score)now_alpha = score;
			}
			if (bestscore < score) {
				bestscore = score;
			}
		}
	}

	if (n_disc < OPTIMAL_TT_THRESHOLD) {
		if (node_kind == REOPENING_AB_PV_NODE || node_kind == REOPENING_AB_ALL_DASH_NODE) {
			assert(alpha == -36 && beta == 36);
		}
		if (alpha < bestscore && bestscore < beta) {
			b.update(bestscore, bestscore);
		}
		else if (bestscore <= alpha) {
			b.update(-36, bestscore);
		}
		else assert(false);
		optimal_reopening_ab_transposition_table[unique_code] = std::make_pair(b, tt_node_kind | uint8_t(node_kind));
	}

	return bestscore;


}

void try_negascout() {
	uint8_t chosen_move = 0xFF;

	const auto initial_position = get_initial_position();
	const auto start = std::chrono::system_clock::now();
	const auto score = negascout(initial_position[0], initial_position[1], -36, 36, NEGASCOUT_PV, chosen_move);
	const auto end = std::chrono::system_clock::now();
	const double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	std::cout << "elapsed time = " << elapsed << " sec" << std::endl;
	std::cout << "score = " << score << std::endl;
	std::cout << "transposition_table.size() = " << transposition_table.size() << std::endl;
	std::cout << "transposition_table.get_bitlen() = " << transposition_table.get_bitlen() << std::endl;
	std::cout << "TT_THRESHOLD = " << TT_THRESHOLD << std::endl;
	std::cout << "ENDGAME_THRESHOLD = " << ENDGAME_THRESHOLD << std::endl;

	//以下はwipeoutをtableに書かない古いバージョンのデータ
	/*
	TT_THRESHOLD=26, ENDGAME_THRESHOLD=6
	score = -4
	transposition_table.size() = 106083179
	transposition_table.get_bitlen() = 28
	70分くらい
	*/

	return;
}

void try_optimal_alphabeta(const std::array<uint64_t, 2> initial_position, const std::string obf67, const int node_kind, const int lowerbound, const int upperbound) {

	const auto start = std::chrono::system_clock::now();

	if (node_kind & OPTIMAL_AB_PV_NODE) {
		optimal_alphabeta(initial_position[0], initial_position[1], -36, 36, OPTIMAL_AB_PV_NODE);
	}
	if (node_kind & OPTIMAL_AB_ALL_NODE) {
		const int alpha = std::min(36, upperbound + 1);
		const int beta = alpha;
		optimal_alphabeta(initial_position[0], initial_position[1], alpha, beta, OPTIMAL_AB_ALL_NODE);
	}
	if (node_kind & OPTIMAL_AB_CUT_NODE) {
		const int alpha = std::max(-36, lowerbound - 1);
		const int beta = alpha;
		optimal_alphabeta(initial_position[0], initial_position[1], alpha, beta, OPTIMAL_AB_CUT_NODE);
	}

	const auto end = std::chrono::system_clock::now();
	const double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	std::cout << "elapsed time = " << elapsed << " sec" << std::endl;
	//std::cout << "score = " << score << std::endl;
	std::cout << "optimal_ab_transposition_table.size() = " << optimal_ab_transposition_table.size() << std::endl;
	std::cout << "transposition_table.size() = " << transposition_table.size() << std::endl;
	std::cout << "transposition_table.get_bitlen() = " << transposition_table.get_bitlen() << std::endl;
	std::cout << "TT_THRESHOLD = " << TT_THRESHOLD << std::endl;
	std::cout << "OPTIMAL_TT_THRESHOLD = " << OPTIMAL_TT_THRESHOLD << std::endl;
	std::cout << "ENDGAME_THRESHOLD = " << ENDGAME_THRESHOLD << std::endl;

	int64_t num_disc_count[37] = {};
	for (const auto e : optimal_ab_transposition_table) {
		const auto bb = decode_bb(e.first);
		assert((bb[0] & bb[1]) == 0);
		++num_disc_count[_mm_popcnt_u64(bb[0] | bb[1])];
	}
	for (int i = 4; i <= 36; ++i) {
		std::cout << i << ", " << num_disc_count[i] << std::endl;
	}

	//optimal_ab_transposition_tableをファイル出力する。
	{
		std::ofstream writing_file;
		const std::string filename = std::string("optimal_ab_transposition_table") + obf67.substr(0, 64) + std::string(".csv");
		const std::string tmp_filename = filename + std::string(".tmp");
		writing_file.open(tmp_filename, std::ios::out);
		writing_file << "obf,disccount,nodekindcode,lowerbound,upperbound" << std::endl;
		for (const auto e : optimal_ab_transposition_table) {
			const std::array<uint64_t, 2> bb = board_unique(decode_bb(e.first));
			assert((bb[0] & bb[1]) == 0);
			const uint8_t nodekindcode = e.second.second;
			const int8_t lb = e.second.first.lowerbound;
			const int8_t ub = e.second.first.upperbound;
			writing_file <<
				bb_to_obf67(bb) << "," <<
				std::to_string(int(_mm_popcnt_u64(bb[0] | bb[1]))) << "," <<
				std::to_string(int(nodekindcode)) << "," <<
				std::to_string(int(lb)) << "," <<
				std::to_string(int(ub)) << std::endl;
		}
		writing_file.close();
		std::filesystem::rename(tmp_filename, filename);
	}

}


void try_optimal_reopening_alphabeta(const std::array<uint64_t, 2> initial_position, const std::string obf67, const int node_kind, const int lowerbound, const int upperbound) {

	const auto start = std::chrono::system_clock::now();

	if (node_kind & REOPENING_AB_PV_NODE) {
		optimal_reopening_alphabeta(initial_position[0], initial_position[1], -36, 36, REOPENING_AB_PV_NODE);
	}
	if (node_kind & REOPENING_AB_ALL_DASH_NODE) {
		optimal_reopening_alphabeta(initial_position[0], initial_position[1], -36, 36, REOPENING_AB_ALL_DASH_NODE);
	}
	if (node_kind & REOPENING_AB_PV_DASH_NODE) {
		optimal_reopening_alphabeta(initial_position[0], initial_position[1], -36, 36, REOPENING_AB_PV_DASH_NODE);
	}
	if (node_kind & REOPENING_AB_ALL_NODE) {
		const int alpha = std::min(36, upperbound + 1);
		const int beta = alpha;
		optimal_reopening_alphabeta(initial_position[0], initial_position[1], alpha, beta, REOPENING_AB_ALL_NODE);
	}
	if (node_kind & REOPENING_AB_CUT_NODE) {
		const int alpha = std::max(-36, lowerbound - 1);
		const int beta = alpha;
		optimal_reopening_alphabeta(initial_position[0], initial_position[1], alpha, beta, REOPENING_AB_CUT_NODE);
	}

	//const auto score = optimal_reopening_alphabeta(initial_position[0], initial_position[1], alpha, beta, node_kind);

	const auto end = std::chrono::system_clock::now();
	const double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	std::cout << "elapsed time = " << elapsed << " sec" << std::endl;
	//std::cout << "score = " << score << std::endl;
	std::cout << "optimal_reopening_ab_transposition_table.size() = " << optimal_reopening_ab_transposition_table.size() << std::endl;
	std::cout << "transposition_table.size() = " << transposition_table.size() << std::endl;
	std::cout << "transposition_table.get_bitlen() = " << transposition_table.get_bitlen() << std::endl;
	std::cout << "TT_THRESHOLD = " << TT_THRESHOLD << std::endl;
	std::cout << "OPTIMAL_TT_THRESHOLD = " << OPTIMAL_TT_THRESHOLD << std::endl;
	std::cout << "ENDGAME_THRESHOLD = " << ENDGAME_THRESHOLD << std::endl;

	int64_t num_disc_count[37] = {};
	for (const auto e : optimal_reopening_ab_transposition_table) {
		const auto bb = decode_bb(e.first);
		assert((bb[0] & bb[1]) == 0);
		++num_disc_count[_mm_popcnt_u64(bb[0] | bb[1])];
	}
	for (int i = 4; i <= 36; ++i) {
		std::cout << i << ", " << num_disc_count[i] << std::endl;
	}

	//optimal_reopening_ab_transposition_tableをファイル出力する。
	{
		std::ofstream writing_file;
		const std::string filename = std::string("optimal_reopening_ab_transposition_table") + obf67.substr(0, 64) + std::string(".csv");
		const std::string tmp_filename = filename + std::string(".tmp");
		writing_file.open(tmp_filename, std::ios::out);
		writing_file << "obf,disccount,nodekindcode,lowerbound,upperbound" << std::endl;
		for (const auto e : optimal_reopening_ab_transposition_table) {
			const std::array<uint64_t, 2> bb = board_unique(decode_bb(e.first));
			assert((bb[0] & bb[1]) == 0);
			const uint8_t nodekindcode = e.second.second;
			const int8_t lb = e.second.first.lowerbound;
			const int8_t ub = e.second.first.upperbound;
			writing_file <<
				bb_to_obf67(bb) << "," <<
				std::to_string(int(_mm_popcnt_u64(bb[0] | bb[1]))) << "," <<
				std::to_string(int(nodekindcode)) << "," <<
				std::to_string(int(lb)) << "," <<
				std::to_string(int(ub)) << std::endl;
		}
		writing_file.close();
		std::filesystem::rename(tmp_filename, filename);
	}

	// //transposition_tableをファイル出力する。
	// if (ENDGAME_THRESHOLD != -1){
	// 	std::ofstream writing_file;
	// 	const std::string filename = std::string("transposition_table") + obf67.substr(0, 64) + std::string(".csv");
	// 	const std::string tmp_filename = filename + std::string(".tmp");
	// 	writing_file.open(tmp_filename, std::ios::out);
	// 	writing_file << "obf,disccount,lowerbound,upperbound" << std::endl;
	// 	for (uint64_t i = 0; i < transposition_table.hash_table.size(); ++i) {
	// 		if (transposition_table.signature_table[i] == 0x80)continue;
	// 		const HashEntry e = transposition_table.hash_table[i];
	// 		const std::array<uint64_t, 2> bb = board_unique(decode_bb(e.encoded_bb));
	// 		assert((bb[0] & bb[1]) == 0);
	// 		const int8_t lb = e.bound.lowerbound;
	// 		const int8_t ub = e.bound.upperbound;
	// 		writing_file <<
	// 			bb_to_obf67(bb) << "," <<
	// 			std::to_string(int(_mm_popcnt_u64(bb[0] | bb[1]))) << "," <<
	// 			std::to_string(int(lb)) << "," <<
	// 			std::to_string(int(ub)) << std::endl;
	// 	}
	// 	writing_file.close();
	// 	std::filesystem::rename(tmp_filename, filename);
	// }
}



template<bool is_reopening>void postprocess1(const std::string input_path, const std::string output_path, const std::string unique_output_suffix) {
	//input_pathがディレクトリのpathを指しているとする。そこにあるファイル名のうち対象となるものを全て取得する。
	std::vector<std::string> filenames;
	const std::regex csv_filetype(is_reopening ? R"(optimal_reopening_ab_transposition_table[-OX]{64}\.csv)" : R"(optimal_ab_transposition_table[-OX]{64}\.csv)");
	for (const auto& entry : std::filesystem::directory_iterator(input_path)) {
		const std::string filename = entry.path().filename().string();
		if (!std::regex_match(filename, csv_filetype)) continue;
		filenames.push_back(filename);
	}

	//例：
	//obf,disccount,nodekindcode,lowerbound,upperbound
	//---------------------------OX------XO--------------------------- X;,4,1,-4,-4
	const std::regex data_linetype(R"(([-OX]{64}\s[OX];),(\d+),(\d+),(-?\d+),(-?\d+))");
	std::vector<std::vector<std::pair<uint64_t, std::array<int16_t, 3>>>> data(37);
	for (const auto filename : filenames) {
		std::vector<std::string> lines;
		//ファイルを読み込んで1行ずつのデータを取得する。
		{
			std::ifstream reading_file;
			reading_file.open(input_path + std::string("/") + filename, std::ios::in);
			std::string line;
			std::getline(reading_file, line);
			while (std::getline(reading_file, line)) {
				lines.push_back(line);
			}
			reading_file.close();
		}

		//data_linetypeにマッチする行について、要素たちを取り出す。

		// std::smatch match;
		// for (const std::string line: lines){
		// 	if(!std::regex_match(line, match, data_linetype)) continue;
		// 	const std::array<uint64_t, 2> bb = board_unique(obf67_to_bb(match[1].str()));
		// 	const auto disccount = _mm_popcnt_u64(bb[0] | bb[1]);
		// 	assert(disccount == std::stoull(match[2].str()));
		// 	data[disccount].push_back(std::make_pair(
		// 		encode_bb(obf67_to_bb(match[1].str())),
		// 		std::array<int16_t, 3>{
		// 			int16_t(std::stoi(match[3].str())),
		// 			int16_t(std::stoi(match[4].str())),
		// 			int16_t(std::stoi(match[5].str()))//note: std::from_charsで高速化できるかもしれない。
		// 		}
		// 	));
		// 	assert(0 < data[disccount].back().second[0] && data[disccount].back().second[0] < 32);
		// 	assert(-36 <= data[disccount].back().second[1] && data[disccount].back().second[1] <= 36);
		// 	assert(-36 <= data[disccount].back().second[2] && data[disccount].back().second[2] <= 36);
		// }

		//上のコードは遅いので、以下のコードに置き換える。lines[0]にはヘッダーが入っているので、それを飛ばす。
		char board_state[68] = {};
		for (size_t i = 0; i < lines.size(); ++i) {
			const char* cstr = lines[i].c_str();
			const char* pattern = cstr;

			// ボードの状態を読み取る
			strncpy(board_state, pattern, 67);
			pattern += 68; // obf + コンマ

			// ディスクカウントを読み取る
			int disccount, nodekindcode, lowerbound, upperbound;

			//patternの次がコンマなら1桁、そうでないなら2桁の数字が続くので、disccountに代入する。
			if (*(pattern + 1) == ',') {
				disccount = *pattern - '0';
				pattern += 2;
			}
			else if (*(pattern + 2) == ',') {
				disccount = (*pattern - '0') * 10 + *(pattern + 1) - '0';
				pattern += 3;
			}
			else assert(false);

			if (*(pattern + 1) == ',') {
				nodekindcode = *pattern - '0';
				pattern += 2;
			}
			else if (*(pattern + 2) == ',') {
				nodekindcode = (*pattern - '0') * 10 + *(pattern + 1) - '0';
				pattern += 3;
			}
			else assert(false);

			if (*pattern == '-') {
				if (*(pattern + 2) == ',') {
					lowerbound = -(*(pattern + 1) - '0');
					pattern += 3;
				}
				else if (*(pattern + 3) == ',') {
					lowerbound = -((*(pattern + 1) - '0') * 10 + *(pattern + 2) - '0');
					pattern += 4;
				}
				else assert(false);
			}
			else if (*(pattern + 1) == ',') {
				lowerbound = *pattern - '0';
				pattern += 2;
			}
			else if (*(pattern + 2) == ',') {
				lowerbound = (*pattern - '0') * 10 + *(pattern + 1) - '0';
				pattern += 3;
			}
			else assert(false);

			if (*pattern == '-') {
				if (*(pattern + 2) == '\0') {
					upperbound = -(*(pattern + 1) - '0');
				}
				else if (*(pattern + 3) == '\0') {
					upperbound = -((*(pattern + 1) - '0') * 10 + *(pattern + 2) - '0');
				}
				else assert(false);
			}
			else if (*(pattern + 1) == '\0') {
				upperbound = *pattern - '0';
			}
			else if (*(pattern + 2) == '\0') {
				upperbound = (*pattern - '0') * 10 + *(pattern + 1) - '0';
			}
			else assert(false);

			// board_stateとして得た情報をbitboardに変換
			std::array<uint64_t, 2> bb = obf67_to_bb(board_state);
			uint64_t encoded = encode_bb(board_unique(obf67_to_bb(board_state)));

			// データを追加
			data[disccount].push_back(
				std::make_pair(
					encoded,
					std::array<int16_t, 3>{
				static_cast<int16_t>(nodekindcode),
					static_cast<int16_t>(lowerbound),
					static_cast<int16_t>(upperbound)
			}
			)
			);
		}

	}

	for (size_t n_disc = 0; n_disc < data.size(); ++n_disc) {
		if (data[n_disc].empty()) continue;

		std::sort(data[n_disc].begin(), data[n_disc].end(), [](const std::pair<uint64_t, std::array<int16_t, 3>>& a, const std::pair<uint64_t, std::array<int16_t, 3>>& b) {
			return a.first < b.first;
			});

		//重複除去の準備をする。重複範囲をみつけたら、いい感じにマージしてから全要素のsecondをその値にする。
		for (size_t i = 0; i < data[n_disc].size() - 1; ++i) {
			if (data[n_disc][i].first != data[n_disc][i + 1].first) continue;
			size_t j = i + 1;
			while (j < data[n_disc].size() && data[n_disc][i].first == data[n_disc][j].first) ++j;

			//この時点で、data[n_disc][i]からdata[n_disc][j-1]までが重複している。

			int16_t nodekindcode = 0, lowerbound = -36, upperbound = 36;
			for (auto k = i; k < j; ++k) {
				nodekindcode |= data[n_disc][k].second[0];
				lowerbound = std::max(lowerbound, data[n_disc][k].second[1]);
				upperbound = std::min(upperbound, data[n_disc][k].second[2]);
			}
			assert(lowerbound <= upperbound);
			if(is_reopening){
				if (nodekindcode & (REOPENING_AB_PV_NODE | REOPENING_AB_ALL_DASH_NODE | REOPENING_AB_PV_DASH_NODE)) {
					assert(lowerbound == upperbound);
				}
			}
			else {
				if (nodekindcode & OPTIMAL_AB_PV_NODE) {
					assert(lowerbound == upperbound);
				}
			}

			for (auto k = i; k < j; ++k) {
				data[n_disc][k].second[0] = nodekindcode;
				data[n_disc][k].second[1] = lowerbound;
				data[n_disc][k].second[2] = upperbound;
			}
			i = j - 1;
		}

		data[n_disc].erase(std::unique(data[n_disc].begin(), data[n_disc].end()), data[n_disc].end());
		//data[n_disc]をテキストファイル出力する。
		{
			std::ofstream writing_file;
			const std::string filename =
				output_path + std::string("/") +
				std::string(is_reopening ? "optimal_reopening_ab_table_" : "optimal_ab_table_") +
				std::to_string(n_disc) + std::string("_") + unique_output_suffix + std::string(".txt");
			const std::string tmp_filename = filename + std::string(".tmp");
			writing_file.open(tmp_filename, std::ios::out);
			char oneline[16] = {};
			oneline[12] = '\n';
			oneline[13] = '\0';
			for (const auto e : data[n_disc]) {
				std::array<char, 9> bb_text = encode_string_bb(decode_bb(e.first));
				for (int j = 0; j < 9; ++j) oneline[j] = bb_text[j];
				oneline[9] = codebook[e.second[0]];
				oneline[10] = codebook[36 + e.second[1]];
				oneline[11] = codebook[36 + e.second[2]];
				oneline[12] = '\n';
				oneline[13] = '\0';
				writing_file << oneline;
			}
			writing_file.close();
			std::filesystem::rename(tmp_filename, filename);
		}
	}
}

template<bool is_reopening>void postprocess3_check_sorted_and_dedup_board(const std::string input_filepath, const std::string output_filepath) {

	//ファイルの各行が何らかの基準でソートされていると仮定する。結果として同じ盤面が連続していることも仮定する。

	//ファイルのバイト数を取得して行数を計算する。
	const std::filesystem::path path(input_filepath);
	const uint64_t filesize = std::filesystem::file_size(path);
	assert(filesize % 13 == 0);
	const uint64_t n_lines = filesize / 13;

	//ファイルを読み込んで1行ずつのデータを取得する。重複をみつけて、ユニークな盤面の数を取得する。
	uint64_t n_unique_boards = 0;
	{
		std::ifstream reading_file;
		reading_file.open(input_filepath, std::ios::in);
		std::string line;
		std::array<char, 9> board;
		uint64_t bb_prev = 0;
		for (uint64_t i = 0; i < n_lines; ++i) {
			std::getline(reading_file, line);
			assert(line.size() == 12);
			for (int j = 0; j < 9; ++j) {
				board[j] = line[j];
			}
			const uint64_t bb_current = encode_bb(decode_string_bb(board));
			if (i == 0 || bb_current != bb_prev) ++n_unique_boards;
			bb_prev = bb_current;
		}
		reading_file.close();
	}

	std::vector<std::pair<uint64_t, std::array<int8_t, 3>>> data(n_unique_boards);

	//ファイルを読み込んで1行ずつのデータを取得する。重複をみつけたらいい感じにマージする。
	{
		std::ifstream reading_file;
		reading_file.open(input_filepath, std::ios::in);
		std::string line;
		std::array<char, 9> board;

		uint64_t cursor = 0, bb_prev = 0;
		for (uint64_t i = 0; i < n_lines; ++i) {
			std::getline(reading_file, line);
			assert(line.size() == 12);
			for (int j = 0; j < 9; ++j) {
				board[j] = line[j];
			}
			const uint64_t bb_current = encode_bb(decode_string_bb(board));
			if (i == 0 || bb_current != bb_prev) {
				if (i != 0) ++cursor;
				data[cursor].second[0] = int8_t(int(inverse_codebook[line[9]]));
				data[cursor].second[1] = int8_t(int(inverse_codebook[line[10]]) - 36);
				data[cursor].second[2] = int8_t(int(inverse_codebook[line[11]]) - 36);
				data[cursor].first = bb_current;
				assert(0 < data[cursor].second[0] && data[cursor].second[0] < 32);
				assert(-36 <= data[cursor].second[1] && data[cursor].second[1] <= data[cursor].second[2] && data[cursor].second[2] <= 36);
			}
			else {
				data[cursor].second[0] |= int8_t(int(inverse_codebook[line[9]]));
				data[cursor].second[1] = int8_t(std::max(int(data[cursor].second[1]), int(inverse_codebook[line[10]]) - 36));
				data[cursor].second[2] = int8_t(std::min(int(data[cursor].second[2]), int(inverse_codebook[line[11]]) - 36));
				assert(0 < data[cursor].second[0] && data[cursor].second[0] < 32);
				assert(-36 <= data[cursor].second[1] && data[cursor].second[1] <= data[cursor].second[2] && data[cursor].second[2] <= 36);
			}
			bb_prev = bb_current;
		}
		reading_file.close();
		assert(cursor + 1 == n_unique_boards);
	}

	std::sort(data.begin(), data.end(), [](const std::pair<uint64_t, std::array<int8_t, 3>>& a, const std::pair<uint64_t, std::array<int8_t, 3>>& b) {
		return a.first < b.first;
		});

	for (uint64_t i = 0; i < data.size(); ++i) {
		if (i)assert(data[i - 1].first < data[i].first);

		int16_t nodekindcode = 0, lowerbound = -36, upperbound = 36;

		nodekindcode = static_cast<int16_t>(data[i].second[0]);
		lowerbound = static_cast<int16_t>(data[i].second[1]);
		upperbound = static_cast<int16_t>(data[i].second[2]);

		assert(lowerbound <= upperbound);
		if (is_reopening) {
			if (nodekindcode & (REOPENING_AB_PV_NODE | REOPENING_AB_ALL_DASH_NODE | REOPENING_AB_PV_DASH_NODE)) {
				assert(lowerbound == upperbound);
			}
		}
		else {
			if (nodekindcode & OPTIMAL_AB_PV_NODE) {
				assert(lowerbound == upperbound);
			}
		}
	}

	//出力する。
	{
		std::ofstream writing_file;
		const std::string tmp_output_filepath = output_filepath + std::string(".tmp");
		writing_file.open(tmp_output_filepath, std::ios::out);
		char oneline[16] = {};
		oneline[12] = '\n';
		oneline[13] = '\0';
		for (uint64_t i = 0; i < data.size(); ++i) {
			std::array<char, 9> bb_text = encode_string_bb(decode_bb(data[i].first));
			for (int j = 0; j < 9; ++j) oneline[j] = bb_text[j];
			oneline[9] = codebook[data[i].second[0]];
			oneline[10] = codebook[36 + data[i].second[1]];
			oneline[11] = codebook[36 + data[i].second[2]];
			oneline[12] = '\n';
			oneline[13] = '\0';
			writing_file << oneline;
		}
		writing_file.close();
		std::filesystem::rename(tmp_output_filepath, output_filepath);
	}
}

uint64_t binarysearch_board(const std::vector<uint64_t> &after_board, const uint64_t code) {
	//after_boardのなかに、codeがあるか二分探索で探す。あればそのindexを返し、なければ0xFFFF'FFFF'FFFF'FFFFULLを返す。
	int64_t left = -1, right = static_cast<int64_t>(after_board.size());
	while (left + 1 < right) {
		const int64_t mid = (left + right) / 2;
		if (after_board[mid] == code) return mid;
		if (after_board[mid] < code) left = mid;
		else right = mid;
	}
	if (0 <= right && right < static_cast<int64_t>(after_board.size()) && after_board[right] == code) return right;
	if (0 <= left && left < static_cast<int64_t>(after_board.size()) && after_board[left] == code) return left;
	return 0xFFFF'FFFF'FFFF'FFFFULL;
}

template<bool is_reopening>void postprocess4_check_consistency(const std::string before_filepath, const std::string after_filepath, const std::string output_filepath) {
	// before_filepathにはpostprocess3まで完了したf"optimal_reopening_ab_table_all_{n}.txt"のファイルパスが入っているとする。
	// after_filepathにはpostprocess3まで完了したf"optimal_reopening_ab_table_all_{n+1}.txt"のファイルパスもしくは""が入っているとする。
	// before_filepath上のすべての盤面の情報について、1手読みしてafter_filepathの情報と照合したときに整合していることを確認する。
	//
	//物理メモリ128GB, 論理CPU数32の環境を想定する。
	//after側は二分探索したいので全情報を物理メモリに乗せる必要があるが、最大7,372,010,761通り(32石)あり、その場合81,092,118,371Bを要する。
	//before側を一気にロードするのではなくチャンクごとにロードすればよい。

	uint64_t after_n_lines = 0;
	if (after_filepath.size() > 0) {
		const std::filesystem::path after_path(after_filepath);
		const uint64_t filesize = std::filesystem::file_size(after_path);
		assert(filesize > 0 && filesize % 13 == 0);
		after_n_lines = filesize / 13;
	}
	std::vector<uint64_t> after_board(after_n_lines);
	std::vector<int8_t> after_data(after_n_lines * 3);
	if (after_filepath.size() > 0) {
		std::ifstream reading_file;
		reading_file.open(after_filepath, std::ios::in);
		std::string line;
		std::array<char, 9> board;
		for (uint64_t i = 0; i < after_n_lines; ++i) {
			std::getline(reading_file, line);
			assert(line.size() == 12);
			for (int j = 0; j < 9; ++j) {
				board[j] = line[j];
			}
			after_board[i] = encode_bb(decode_string_bb(board));
			after_data[i * 3 + 0] = int8_t(int(inverse_codebook[line[9]]));
			after_data[i * 3 + 1] = int8_t(int(inverse_codebook[line[10]]) - 36);
			after_data[i * 3 + 2] = int8_t(int(inverse_codebook[line[11]]) - 36);
			if (i > 0)assert(after_board[i - 1] < after_board[i]);
			assert(0 < after_data[i * 3] && after_data[i * 3] < 32);
			assert(-36 <= after_data[i * 3 + 1] && after_data[i * 3 + 1] <= 36);
			assert(-36 <= after_data[i * 3 + 2] && after_data[i * 3 + 2] <= 36);
		}
		reading_file.close();
	}
	uint64_t before_n_lines = 0;
	{
		assert(before_filepath.size() > 0);
		const std::filesystem::path before_path(before_filepath);
		const uint64_t filesize = std::filesystem::file_size(before_path);
		assert(filesize > 0 && filesize % 13 == 0);
		before_n_lines = filesize / 13;
	}

	std::ifstream reading_file;
	reading_file.open(before_filepath, std::ios::in);
	uint64_t read_count = 0;
	constexpr uint64_t chunk_size = 640;

	/*
	考察

	チェックすべき事柄一覧：

	beforeがREOPENING_AB_PV_NODEの場合:
	(1) bestがPVかobvious
	(2) それ以外全てがALL_DASHかobviousか   //6*6オセロでPV_NODEのスコアが36になることは決してないのでafterは必ずヒットする
	(3) 全てがlowerbound==upperboundである
	(4) best⇔beforeとafterのlowerboundとupperboundが一致する。
	beforeがREOPENING_AB_ALL_DASH_NODEの場合:
	(5) bestがALL_DASHかobviousか上位互換（PV）
	(6) （チェック不要）best以外のnode_kindはなんでもいい
	(7) bestがlowerbound==upperboundである
	(8) best⇔beforeとafterのlowerboundとupperboundが一致する。
	beforeがREOPENING_AB_PV_DASH_NODEの場合:
	(9) afterの全てがALL_DASHかobviousか上位互換（PV）   //6*6オセロでPV_DASH_NODEのスコアが36になることは決してないのでafterは必ずヒットする
	(10) 全てがlowerbound==upperboundである
	(11) best⇔beforeとafterのlowerboundとupperboundが一致する。
	beforeがREOPENING_AB_ALL_NODEの場合:
	(12) 全てがCUTかALLかobviousか上位互換（lowerbound==upperbound）  //CUT,ALLにおいては、transposition table上で値が一意なものならなんでも使ってreturnする。
	(13) beforeのupperbound（=afterのlowerbound）だけ確認すればいい。
	beforeがREOPENING_AB_CUT_NODEの場合:
	(14) 一つ以上存在して、CUTかALLかobviousか上位互換（lowerbound==upperbound）  //CUT,ALLにおいては、transposition table上で値が一意なものならなんでも使ってreturnする。
	(15) beforeのlowerbound（=afterのupperbound）だけ確認すればいい。
	(16) afterがヒットしないのはbeforeがCUT_NODEの場合だけ

	この処理はopenmpで並列化できる。

	*/

	const int64_t n_chunk = (before_n_lines + chunk_size - 1) / chunk_size;

	int64_t process_count = 0;
#pragma omp parallel for
	for (int64_t i = 0; i < n_chunk; ++i) {
		std::vector<uint64_t> before_board;
		std::vector<int8_t> before_data;
#pragma omp critical
		{
			const uint64_t read_size = std::min(chunk_size, before_n_lines - read_count);
			before_board.resize(read_size);
			before_data.resize(read_size * 3);
			std::cout << "chunk " << process_count++ << " / " << n_chunk << std::endl;
			//lambda_reader(before_board, before_data);
			std::string line;
			std::array<char, 9> board;
			uint64_t j = 0;
			for (; j < chunk_size && read_count < before_n_lines; ++j, ++read_count) {
				std::getline(reading_file, line);
				assert(line.size() == 12);
				for (int j = 0; j < 9; ++j) {
					board[j] = line[j];
				}
				const uint64_t bb_current = encode_bb(decode_string_bb(board));
				const int8_t nodekindcode = int8_t(int(inverse_codebook[line[9]]));
				const int8_t lowerbound = int8_t(int(inverse_codebook[line[10]]) - 36);
				const int8_t upperbound = int8_t(int(inverse_codebook[line[11]]) - 36);
				assert(0 < nodekindcode && nodekindcode < 32);
				assert(-36 <= lowerbound && lowerbound <= 36);
				assert(-36 <= upperbound && upperbound <= 36);
				before_board[j] = bb_current;
				before_data[j * 3 + 0] = nodekindcode;
				before_data[j * 3 + 1] = lowerbound;
				before_data[j * 3 + 2] = upperbound;
			}
			assert(j == before_board.size());
		}
		for (uint64_t j = 0; j < before_board.size(); ++j) {
			std::array<uint64_t, 2> bb = decode_bb(before_board[j]);
			const uint64_t player = bb[0];
			const uint64_t opponent = bb[1];
			const int32_t nodekindcode_before = before_data[j * 3 + 0];
			const int32_t lowerbound_before_in_table = before_data[j * 3 + 1];
			const int32_t upperbound_before_in_table = before_data[j * 3 + 2];

			if(is_reopening){
				if (nodekindcode_before & (REOPENING_AB_PV_NODE | REOPENING_AB_ALL_DASH_NODE | REOPENING_AB_PV_DASH_NODE)) {
					assert(lowerbound_before_in_table == upperbound_before_in_table);//(3),(7),(10)
				}
			}
			else {
				if (nodekindcode_before & OPTIMAL_AB_PV_NODE) {
					assert(lowerbound_before_in_table == upperbound_before_in_table);
				}
			}

			const uint64_t bb_moves = get_moves(player, opponent) & MASK_8x8to6x6;
			assert(bb_moves != 0);

			uint64_t bb_ = bb_moves;
			int32_t best_observed_after_lowerbound = -999;
			int32_t best_observed_after_upperbound = -999;
			int32_t best_obvious_score = -999;
			bool is_best_found = false;
			for (uint32_t index; bitscan_forward64(bb_, &index); bb_ &= bb_ - 1) {

				const uint64_t flipped = flip(player, opponent, index);
				assert(flipped);
				uint64_t next_player = opponent ^ flipped;
				uint64_t next_opponent = player ^ (flipped | (1ULL << index));
				uint64_t next_bb_moves = get_moves(next_player, next_opponent) & MASK_8x8to6x6;
				int32_t pass_coef = 1;

				//パスを考慮する。自明（終局）かどうか調べる。
				if (next_bb_moves == 0) {
					if ((get_moves(next_opponent, next_player) & MASK_8x8to6x6) != 0) { // pass
						std::swap(next_player, next_opponent);
						next_bb_moves = get_moves(next_player, next_opponent) & MASK_8x8to6x6;
						pass_coef = -1;
					}
					else { // game over
						const int32_t current_score = -ComputeFinalScore(next_player, next_opponent);
						best_obvious_score = std::max(best_obvious_score, current_score);
						best_observed_after_lowerbound = std::max(best_observed_after_lowerbound, current_score);
						best_observed_after_upperbound = std::max(best_observed_after_upperbound, current_score);
						continue;
					}
				}

				//自明でないなら、after_boardにあるはず。(直前がCUT NODEなら別。あるいはbeforeのスコアが+36ならafterのALLとCUTは省略されるので別)
				const uint64_t next_bb = encode_bb(board_unique({ next_player, next_opponent }));
				const uint64_t index_in_after = binarysearch_board(after_board, next_bb);
				if (index_in_after == 0xFFFF'FFFF'FFFF'FFFFULL) {
					if(is_reopening){
						
						if(!(nodekindcode_before == REOPENING_AB_CUT_NODE || lowerbound_before_in_table == 36)){
							std::cout << bb_to_obf67(bb) << std::endl;
							print_bb(bb);
							std::cout << nodekindcode_before << " " << lowerbound_before_in_table << " " << upperbound_before_in_table << std::endl;
						}
						assert(nodekindcode_before == REOPENING_AB_CUT_NODE || lowerbound_before_in_table == 36);//(6),(16)
						if (lowerbound_before_in_table == 36)assert((nodekindcode_before & (REOPENING_AB_PV_NODE | REOPENING_AB_PV_DASH_NODE)) == 0);//(2),(9)
					}
					else{
						assert(nodekindcode_before == OPTIMAL_AB_CUT_NODE || lowerbound_before_in_table == 36);
						if (lowerbound_before_in_table == 36)assert((nodekindcode_before & OPTIMAL_AB_PV_NODE) == 0);
					}
					continue;
				}
				int32_t nodekindcode_after = 0;
				if (pass_coef == 1)nodekindcode_after = after_data[index_in_after * 3 + 0];
				else {
					for (int bit = 1; bit < 32; bit <<= 1) {
						if (after_data[index_in_after * 3 + 0] & bit){
							nodekindcode_after |= is_reopening ? get_reopening_ab_child_node_kind(bit, true) : get_optimal_ab_child_node_kind(bit, true);
						}
					}
				}
				const int32_t lowerbound_after_in_table = -pass_coef * after_data[index_in_after * 3 + (pass_coef == 1 ? 2 : 1)];
				const int32_t upperbound_after_in_table = -pass_coef * after_data[index_in_after * 3 + (pass_coef == 1 ? 1 : 2)];

				if(is_reopening){
					if (nodekindcode_before & (REOPENING_AB_PV_NODE | REOPENING_AB_ALL_DASH_NODE | REOPENING_AB_PV_DASH_NODE)) {
						static_assert(chunk_size % 64 == 0);//This condition guarantees that atomic operation is not needed.
						if (nodekindcode_before & REOPENING_AB_PV_NODE) {
							if (lowerbound_before_in_table == lowerbound_after_in_table && (nodekindcode_after & REOPENING_AB_PV_NODE))is_best_found = true;//(1),(4)
							assert(nodekindcode_after & (REOPENING_AB_ALL_DASH_NODE | REOPENING_AB_PV_NODE));//(2)
							assert(lowerbound_after_in_table == upperbound_after_in_table);//(3)
						}
						if (nodekindcode_before & REOPENING_AB_ALL_DASH_NODE) {
							if (lowerbound_before_in_table == lowerbound_after_in_table &&
								(nodekindcode_after & (REOPENING_AB_PV_DASH_NODE | REOPENING_AB_PV_NODE)))is_best_found = true;//(5),(7),(8)
						}
						if (nodekindcode_before & REOPENING_AB_PV_DASH_NODE) {
							if (lowerbound_before_in_table == lowerbound_after_in_table &&
								(nodekindcode_after & (REOPENING_AB_ALL_DASH_NODE | REOPENING_AB_PV_NODE)))is_best_found = true;//(9)
							assert(nodekindcode_after & (REOPENING_AB_ALL_DASH_NODE | REOPENING_AB_PV_NODE));//((11)
							assert(lowerbound_after_in_table == upperbound_after_in_table);//(10)
						}
					}
				}
				else{
					if (nodekindcode_before & OPTIMAL_AB_PV_NODE) {
						static_assert(chunk_size % 64 == 0);//This condition guarantees that atomic operation is not needed.
						if (lowerbound_before_in_table == lowerbound_after_in_table && (nodekindcode_after & OPTIMAL_AB_PV_NODE))is_best_found = true;
						assert(nodekindcode_after & (OPTIMAL_AB_CUT_NODE | OPTIMAL_AB_PV_NODE));
					}
				}
				best_observed_after_lowerbound = std::max(best_observed_after_lowerbound, upperbound_after_in_table);
				best_observed_after_upperbound = std::max(best_observed_after_upperbound, lowerbound_after_in_table);
			}

			if (nodekindcode_before & (is_reopening ? (REOPENING_AB_PV_NODE | REOPENING_AB_ALL_DASH_NODE | REOPENING_AB_PV_DASH_NODE) : OPTIMAL_AB_PV_NODE)) {
				assert(is_best_found || best_obvious_score == best_observed_after_upperbound);//(1),(4),(5),(8),(9),(11)
				assert(lowerbound_before_in_table == best_observed_after_lowerbound
					&& upperbound_before_in_table == best_observed_after_upperbound);//(4),(8),(11). obviousがbestの場合をケアしている
				assert(best_observed_after_lowerbound == best_observed_after_upperbound);//(7)
			}

			if (nodekindcode_before & (is_reopening ? REOPENING_AB_ALL_NODE : OPTIMAL_AB_ALL_NODE)) {
				assert(std::max(best_obvious_score, best_observed_after_upperbound) <= upperbound_before_in_table);//(12),(13). obviousを含めて全てが条件を満たすことを確認している
			}

			if (nodekindcode_before & (is_reopening ? REOPENING_AB_CUT_NODE : OPTIMAL_AB_CUT_NODE)) {
				assert(lowerbound_before_in_table <= std::max(best_obvious_score, best_observed_after_lowerbound));//(14),(15). obviousを含めて全てが条件を満たすことを確認している
			}
		}
	}


	//出力する。
	{
		std::ofstream writing_file;
		const std::string tmp_output_filepath = output_filepath + std::string(".tmp");
		writing_file.open(tmp_output_filepath, std::ios::out);
		writing_file << before_filepath << " -> " << after_filepath << ": consistent." << std::endl;
		writing_file.close();
		std::filesystem::rename(tmp_output_filepath, output_filepath);
	}

	return;
}

int main(int argc, char *argv[]) {

	init_ternary_tables();
	init_solution_unique_boards();



	// 引数をstd::vector<std::string>に変換
	std::vector<std::string> args(argv, argv + argc);

	if (args.size() == 1) {
		TT_THRESHOLD = 24;
		OPTIMAL_TT_THRESHOLD = 11;
		std::cout << "start: optimal_reopening_alphabeta" << std::endl;
		try_optimal_reopening_alphabeta(get_initial_position(), bb_to_obf67(get_initial_position()), REOPENING_AB_PV_NODE, -36, 36);
		return 0;
	}
	else if (args.size() == 2) {
		if (args[1] == "optimal-alphabeta") {
			TT_THRESHOLD = 24;
			OPTIMAL_TT_THRESHOLD = 19;
			std::cout << "start: optimal_alphabeta" << std::endl;
		try_optimal_alphabeta(get_initial_position(), bb_to_obf67(get_initial_position()), OPTIMAL_AB_PV_NODE, -36, 36);
			return 0;
		}
		std::cout << "error:20" << std::endl;
		return 0;
	}
	else if (args.size() == 3) {
		std::cout << "error:63" << std::endl;
		return 0;
	}
	else if (args.size() == 4) {
		if (args[1] == "postprocess3") {
			postprocess3_check_sorted_and_dedup_board<true>(args[2], args[3]);
			return 0;
		}
		if (args[1] == "postprocess4") {
			postprocess4_check_consistency<true>(args[2], std::string(""), args[3]);
			return 0;
		}
		if (args[1] == "optimal-ab-postprocess3") {
			postprocess3_check_sorted_and_dedup_board<false>(args[2], args[3]);
			return 0;
		}
		if (args[1] == "optimal-ab-postprocess4") {
			postprocess4_check_consistency<false>(args[2], std::string(""), args[3]);
			return 0;
		}
		std::cout << "error:63" << std::endl;
		return 0;
	}
	else if (args.size() == 5) {
		if (args[1] == "postprocess1") {
			postprocess1<true>(args[2], args[3], args[4]);
			return 0;
		}
		if (args[1] == "postprocess4") {
			postprocess4_check_consistency<true>(args[2], args[3], args[4]);
			return 0;
		}
		if (args[1] == "optimal-ab-postprocess1") {
			postprocess1<false>(args[2], args[3], args[4]);
			return 0;
		}
		if (args[1] == "optimal-ab-postprocess4") {
			postprocess4_check_consistency<false>(args[2], args[3], args[4]);
			return 0;
		}
		std::cout << "error:65" << std::endl;
		return 0;
	}
	else if (args.size() == 6 || args.size() == 7) {
		if (args[1] == "reopening-ab") {
			const std::regex positive_integer(R"([1-9][0-9]*)");
			if (!std::regex_match(args[2], positive_integer)) {
				std::cout << "error:41" << std::endl;
				return 0;
			}
			const int kind_number = std::stoi(args[2]);
			if (!(1 <= kind_number && kind_number < 32)) {
				std::cout << "error:kind_number is invalid" << std::endl;
				return 0;
			}

			const int lowerbound = std::stoi(args[3]);
			const int upperbound = std::stoi(args[4]);
			if (!(-36 <= lowerbound && lowerbound <= upperbound && upperbound <= 36)) {
				std::cout << "error:lowerbound upperbound values are invalid" << std::endl;
				return 0;
			}

			const std::string obf67 = (args.size() == 6) ? args[5] : (args[5] + std::string(" ") + args[6]);

			const std::regex obf67_regex(R"([-OX]{64}\s[OX];)");
			if (!std::regex_match(obf67, obf67_regex)) {
				std::cout << "error:42:" << obf67 << std::endl;
				return 0;
			}

			const std::array<uint64_t, 2> starting_position = obf67_to_bb(obf67);
			const auto n_discs = _mm_popcnt_u64(starting_position[0] | starting_position[1]);

			if (n_discs == 10) {
				TT_THRESHOLD = 24;
				OPTIMAL_TT_THRESHOLD = 19;
			}
			else if (n_discs == 18) {
				ENDGAME_THRESHOLD = -1;
				TT_THRESHOLD = 200;
				OPTIMAL_TT_THRESHOLD = 100;
			}
			else {
				std::cout << "error:43" << std::endl;
				return 0;
			}
			std::cout << "start: optimal_reopening_alphabeta" << std::endl;
			try_optimal_reopening_alphabeta(starting_position, obf67, kind_number, lowerbound, upperbound);
			return 0;
		}

		if (args[1] == "optimal-alphabeta") {
			const std::regex positive_integer(R"([1-9][0-9]*)");
			if (!std::regex_match(args[2], positive_integer)) {
				std::cout << "error:41" << std::endl;
				return 0;
			}
			const int kind_number = std::stoi(args[2]);
			if (!(1 <= kind_number && kind_number < 8)) {
				std::cout << "error:kind_number is invalid" << std::endl;
				return 0;
			}

			const int lowerbound = std::stoi(args[3]);
			const int upperbound = std::stoi(args[4]);
			if (!(-36 <= lowerbound && lowerbound <= upperbound && upperbound <= 36)) {
				std::cout << "error:lowerbound upperbound values are invalid" << std::endl;
				return 0;
			}

			const std::string obf67 = (args.size() == 6) ? args[5] : (args[5] + std::string(" ") + args[6]);

			const std::regex obf67_regex(R"([-OX]{64}\s[OX];)");
			if (!std::regex_match(obf67, obf67_regex)) {
				std::cout << "error:42:" << obf67 << std::endl;
				return 0;
			}

			const std::array<uint64_t, 2> starting_position = obf67_to_bb(obf67);
			const auto n_discs = _mm_popcnt_u64(starting_position[0] | starting_position[1]);

			if (n_discs == 10) {
				TT_THRESHOLD = 24;
				OPTIMAL_TT_THRESHOLD = 19;
			}
			else if (n_discs == 18) {
				ENDGAME_THRESHOLD = -1;
				TT_THRESHOLD = 200;
				OPTIMAL_TT_THRESHOLD = 100;
			}
			else {
				std::cout << "error:43" << std::endl;
				return 0;
			}
			std::cout << "start: optimal_alphabeta" << std::endl;
			try_optimal_alphabeta(starting_position, obf67, kind_number, lowerbound, upperbound);
			return 0;
		}

		std::cout << "error:99" << std::endl;
		return 0;
	}

	std::cout << "error:0" << std::endl;
	return 0;
}