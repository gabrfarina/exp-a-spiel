#pragma once

#include <vector>
#include "utils.h"

template <typename T>
class VecEnv {
protected:
    int num_envs;
    std::vector<T> envs;
    std::vector<std::array<std::array<bool, T::OPENSPIEL_INFOSTATE_SIZE>, 2>> tensor;

    auto obs() const {
        return tensor;
    }

    auto current_players() const {
        std::vector<int> buf;
        buf.resize(num_envs);
        for (int i = 0; i < num_envs; i ++) {
            buf[i] = envs[i].player();
        }
        return buf;
    }

    auto action_masks() {
        std::vector<std::array<bool, 9>> buf;
        buf.resize(num_envs);
        for (int i = 0; i < num_envs; i ++) {
            for (int j = 0, a = envs[i].available_actions(); j < 9; j ++, a >>= 1) {
                buf[i][j] = a & 1;
            }
        }
        return buf;
    }

public:
    VecEnv(int num_envs) : num_envs(num_envs) {
        envs.resize(num_envs);
        tensor.resize(num_envs);
    }

    auto reset() {
        for (int i = 0; i < num_envs; i ++) {
            envs[i] = T();
            for (int j = 0; j < 2; j ++)
                T::compute_openspiel_infostate(0, envs[i].get_infoset(), tensor[i][j]);
        }
        return std::make_tuple(obs(), current_players(), action_masks());
    }

    auto step(const std::vector<uint8_t> &actions) {
        std::vector<bool> dones;
        std::vector<std::array<int, 2>> rewards;
        dones.resize(num_envs);
        rewards.resize(num_envs);
        for (int i = 0; i < num_envs; i ++) {

            CHECK(actions[i] < 9, "Invalid cell (must be in range [0..8]; found %d)",
                    cell);
            CHECK(!envs[i].is_terminal(), "Game is over");
            const uint32_t a = envs[i].available_actions();
            CHECK(a & (1 << actions[i]), "The action is not legal");

            envs[i].next(actions[i]);

            if (envs[i].is_terminal()) {
                dones[i] = true;
                const int w = envs[i].winner();
                if (w < 2) {
                    rewards[i][w] = 1;
                    rewards[i][1 - w] = -1;
                } else {
                    assert(w == TIE);
                    rewards[i][w] = 0;
                    rewards[i][1 - w] = 0;
                }
                envs[i] = T();
                for (int j = 0; j < 2; j ++)
                    T::compute_openspiel_infostate(0, envs[i].get_infoset(), tensor[i][j]);
            } else {
                int j = envs[i].player();
                T::compute_openspiel_infostate(j, envs[i].get_infoset(), tensor[i][j]);
            }
        }
        return std::make_tuple(obs(), rewards, dones, current_players(), action_masks());
    }

    void close() { }
};