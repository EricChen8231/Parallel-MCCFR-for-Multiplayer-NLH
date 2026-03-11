#pragma once
#include <cstdint>
#include <string>
#include <random>
#include <algorithm>

// ---------------------------------------------------------------------------
// Card: uint8_t 0-51
//   rank = card % 13   (0=2, 1=3, ..., 12=Ace)
//   suit = card / 13   (0=clubs, 1=diamonds, 2=hearts, 3=spades)
// ---------------------------------------------------------------------------
using Card = uint8_t;

static inline int card_rank(Card c) { return c % 13; }
static inline int card_suit(Card c) { return c / 13; }

static inline std::string card_to_str(Card c) {
    static const char* RANKS = "23456789TJQKA";
    static const char* SUITS = "cdhs";
    std::string s;
    s += RANKS[card_rank(c)];
    s += SUITS[card_suit(c)];
    return s;
}

static inline Card str_to_card(const char* s) {
    static const char* RANKS = "23456789TJQKA";
    static const char* SUITS = "cdhs";
    int r = 0, su = 0;
    for (int i = 0; i < 13; i++) if (RANKS[i] == s[0]) r = i;
    for (int i = 0; i < 4;  i++) if (SUITS[i] == s[1]) su = i;
    return (Card)(su * 13 + r);
}

// ---------------------------------------------------------------------------
// Deck with Fisher-Yates shuffle
// ---------------------------------------------------------------------------
struct Deck {
    Card cards[52];
    int  top = 0;
    Deck() { for (int i = 0; i < 52; i++) cards[i] = (Card)i; }
    void shuffle(std::mt19937& rng) {
        top = 0;
        for (int i = 51; i > 0; i--) {
            int j = (int)(rng() % (i + 1));
            std::swap(cards[i], cards[j]);
        }
    }
    Card deal() { return cards[top++]; }
};
