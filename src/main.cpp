#if defined(ESP_PLATFORM)
#include "idf_compat.hpp"
#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "esp_random.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#else
#include <Arduino.h>
#endif
#include <ctype.h>
#include <string.h>
#if defined(ESP8266) || defined(ESP8266_BOARD)
#  include <ESP8266WiFi.h>
#endif
#include "inference.hpp"
#include "model_weights.hpp"
#include "model_fingerprint.hpp"

static constexpr int N_EMBD          = model_n_embd;
static constexpr int N_HEAD          = model_n_head;
static constexpr int N_KV_HEAD       = model_n_kv_head;
static constexpr int N_LAYER         = model_n_layer;
static constexpr int HEAD_DIM        = N_EMBD / N_HEAD;  
static constexpr int MLP_HIDDEN      = model_mlp_hidden;              
static constexpr int N_EXPERTS       = model_n_experts;                
static constexpr int GRP             = model_group_size;                
#if defined(ESP8266) || defined(ESP8266_BOARD)
static constexpr int INFER_CTX       = 28;
static constexpr int MAX_GEN_TOKENS  = 60;
static constexpr float TEMPERATURE   = 0.0f;
static constexpr size_t ARENA_SIZE   = 40 * 1024;

static uint8_t s_arena_mem[ARENA_SIZE] __attribute__((aligned(4)));
static MemoryArena s_arena(s_arena_mem, ARENA_SIZE);
static MemoryArena* arena = &s_arena;
#elif defined(ESP32S3_BOARD) || defined(ESP_PLATFORM)
// High-capacity profile: ESP32-S3 N16R8 (16MB flash + 8MB PSRAM).
// Arena (4.0MB used) lives in octal PSRAM, supporting up to 512 context tokens.
static constexpr int INFER_CTX       = (512 < (int)model_block_size) ? 512 : (int)model_block_size;
static constexpr int MAX_GEN_TOKENS  = 256;
static constexpr float TEMPERATURE   = 0.0f;
// Match Python Transformer.generate(rep_penalty=1.0) by default.
static constexpr float REPETITION_LOGIT_PENALTY = 0.0f;
static constexpr size_t ARENA_SIZE   = 4096 * 1024;
static MemoryArena* arena = nullptr;
#else
static constexpr int INFER_CTX       = 64;
static constexpr int MAX_GEN_TOKENS  = 80;
static constexpr float TEMPERATURE   = 0.0f;
static constexpr size_t ARENA_SIZE   = 160 * 1024;
static MemoryArena* arena = nullptr;
#endif

static float* g_x;          
static float* g_kbuf;       
static float* g_vbuf;       
static float* g_xnorm;      
static float* g_qkv_out;   
static float* g_attn_out;   
static float* g_proj_out;   
static float* g_att;        
static float* g_mlp_gate;   
static float* g_mlp_up;     
static float* g_mlp_hidden; 
static float* g_mlp_out;    
static float* g_logits;     
static uint16_t ctx_ids[INFER_CTX];
static int     ctx_len = 0;
static int     ctx_pos = 0;

#if defined(ESP32S3_BOARD) || defined(ESP_PLATFORM)
struct ExpertPSRAMCache {
    uint8_t*  gate_q;
    uint16_t* gate_s;
    uint8_t*  up_q;
    uint16_t* up_s;
    uint8_t*  down_q;
    uint16_t* down_s;
    int       cached_layer;
    int       cached_expert;
};
static ExpertPSRAMCache g_expert_cache = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, -1, -1 };
#endif

// Enabled only by :dbgzh1. Keep normal inference free of serial debug overhead.
static bool g_debug_inference = false;

static void print_debug_activation(const char* stage, int layer, const float* values, int n) {
    double sum = 0.0;
    double sumsq = 0.0;
    float maxabs = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float value = values[i];
        sum += value;
        sumsq += (double)value * value;
        const float abs_value = fabsf(value);
        if (abs_value > maxabs) maxabs = abs_value;
    }
    Serial.printf("[DBG]   layer %d %s sum=%.7g sumsq=%.7g maxabs=%.7g\n",
                  layer + 1, stage, sum, sumsq, maxabs);
}

// Inference context must not exceed the RoPE tables baked from training.
static_assert(INFER_CTX <= (int)model_block_size, "INFER_CTX exceeds trained block_size: regenerate model_weights.hpp");

static void transformer_forward(int token, int pos) {
    if (g_debug_inference) {
        Serial.printf("[DBG] prefill token %d/%d (id=%d)\n", pos + 1, ctx_len, token);
    }
    dequant_emb_row(tok_emb_q + (size_t)token * N_EMBD, tok_emb_scales[token], g_x, N_EMBD);
    for (int l = 0; l < N_LAYER; ++l) {
        if (g_debug_inference) Serial.printf("[DBG]   layer %d/%d start\n", l + 1, N_LAYER);
        const LayerW& lw = g_layers[l];
        rms_norm(g_x, g_xnorm, lw.ln1_g, N_EMBD);
        matmul_bitnet_ternary(lw.qkv_w, lw.qkv_s, g_xnorm, g_qkv_out, N_EMBD + 2 * N_KV_HEAD * HEAD_DIM, N_EMBD, GRP);
        float* layer_kbuf = g_kbuf + l * (INFER_CTX * N_KV_HEAD * HEAD_DIM);
        float* layer_vbuf = g_vbuf + l * (INFER_CTX * N_KV_HEAD * HEAD_DIM);
        float* k_pos = layer_kbuf + pos * (N_KV_HEAD * HEAD_DIM);
        float* v_pos = layer_vbuf + pos * (N_KV_HEAD * HEAD_DIM);
        memcpy(k_pos, g_qkv_out + N_EMBD,                        N_KV_HEAD * HEAD_DIM * sizeof(float));
        memcpy(v_pos, g_qkv_out + N_EMBD + N_KV_HEAD * HEAD_DIM, N_KV_HEAD * HEAD_DIM * sizeof(float));
        const float* cos_t = rope_cos + pos * (HEAD_DIM / 2);
        const float* sin_t = rope_sin + pos * (HEAD_DIM / 2);
        for (int h = 0; h < N_HEAD; ++h) {
            apply_rope_row(g_qkv_out + h * HEAD_DIM, cos_t, sin_t, HEAD_DIM);
        }
        for (int h = 0; h < N_KV_HEAD; ++h) {
            apply_rope_row(k_pos + h * HEAD_DIM, cos_t, sin_t, HEAD_DIM);
        }
        memset(g_attn_out, 0, N_EMBD * sizeof(float));
        const float scale = 1.0f / sqrtf((float)HEAD_DIM);
        for (int h = 0; h < N_HEAD; ++h) {
            const float* q_h = g_qkv_out + h * HEAD_DIM;
            int kv_h = h / (N_HEAD / N_KV_HEAD);
            for (int s = 0; s <= pos; ++s)
                g_att[s] = dot_f32(q_h, layer_kbuf + s * (N_KV_HEAD * HEAD_DIM) + kv_h * HEAD_DIM, HEAD_DIM) * scale;
            softmax(g_att, pos + 1);
            float* out_h = g_attn_out + h * HEAD_DIM;
            for (int s = 0; s <= pos; ++s) {
                const float a = g_att[s];
                const float* v_h_ptr = layer_vbuf + s * (N_KV_HEAD * HEAD_DIM) + kv_h * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; ++d) out_h[d] += a * v_h_ptr[d];
            }
        }
        matmul_bitnet_ternary(lw.proj_w, lw.proj_s, g_attn_out, g_proj_out, N_EMBD, N_EMBD, GRP);
        for (int d = 0; d < N_EMBD; ++d) g_x[d] += g_proj_out[d];
        if (g_debug_inference && pos == ctx_len - 1) {
            print_debug_activation("attn", l, g_x, N_EMBD);
        }
        rms_norm(g_x, g_xnorm, lw.ln2_g, N_EMBD);
        int best_expert = 0;
        float best_score = -1e9f;
        for (int e = 0; e < N_EXPERTS; ++e) {
            float score = dot_f32(g_xnorm, lw.router_w + e * N_EMBD, N_EMBD);
            if (score > best_score) {
                best_score = score;
                best_expert = e;
            }
        }
        if (g_debug_inference && pos == ctx_len - 1) {
            Serial.printf("[DBG]   layer %d/%d expert=%d\n", l + 1, N_LAYER, best_expert);
        }
        int gate_n_groups = (N_EMBD + GRP - 1) / GRP;
        int gate_bytes_per_row = gate_n_groups * ((GRP + 4) / 5);
        int gate_q_off = best_expert * MLP_HIDDEN * gate_bytes_per_row;
        int gate_s_off = best_expert * MLP_HIDDEN * gate_n_groups;

        int down_n_groups = (MLP_HIDDEN + GRP - 1) / GRP;
        int down_bytes_per_row = down_n_groups * ((GRP + 4) / 5);
        int down_q_off = best_expert * N_EMBD * down_bytes_per_row;
        int down_s_off = best_expert * N_EMBD * down_n_groups;

        const uint8_t* p_gate_q = lw.experts_gate_q + gate_q_off;
        const uint16_t* p_gate_s = lw.experts_gate_s + gate_s_off;
        const uint8_t* p_up_q = lw.experts_up_q + gate_q_off;
        const uint16_t* p_up_s = lw.experts_up_s + gate_s_off;
        const uint8_t* p_down_q = lw.experts_down_q + down_q_off;
        const uint16_t* p_down_s = lw.experts_down_s + down_s_off;

#if defined(ESP32S3_BOARD) || defined(ESP_PLATFORM)
        if (g_expert_cache.gate_q && (g_expert_cache.cached_layer != l || g_expert_cache.cached_expert != best_expert)) {
            size_t gate_q_sz = (size_t)MLP_HIDDEN * gate_bytes_per_row;
            size_t gate_s_sz = (size_t)MLP_HIDDEN * gate_n_groups * sizeof(uint16_t);
            size_t down_q_sz = (size_t)N_EMBD * down_bytes_per_row;
            size_t down_s_sz = (size_t)N_EMBD * down_n_groups * sizeof(uint16_t);

            memcpy(g_expert_cache.gate_q, p_gate_q, gate_q_sz);
            memcpy(g_expert_cache.gate_s, p_gate_s, gate_s_sz);
            memcpy(g_expert_cache.up_q, p_up_q, gate_q_sz);
            memcpy(g_expert_cache.up_s, p_up_s, gate_s_sz);
            memcpy(g_expert_cache.down_q, p_down_q, down_q_sz);
            memcpy(g_expert_cache.down_s, p_down_s, down_s_sz);

            g_expert_cache.cached_layer = l;
            g_expert_cache.cached_expert = best_expert;
        }
        if (g_expert_cache.gate_q) {
            p_gate_q = g_expert_cache.gate_q;
            p_gate_s = g_expert_cache.gate_s;
            p_up_q   = g_expert_cache.up_q;
            p_up_s   = g_expert_cache.up_s;
            p_down_q = g_expert_cache.down_q;
            p_down_s = g_expert_cache.down_s;
        }
#endif

        matmul_bitnet_ternary(p_gate_q, p_gate_s,
                        g_xnorm, g_mlp_gate, MLP_HIDDEN, N_EMBD, GRP);
        matmul_bitnet_ternary(p_up_q, p_up_s,
                        g_xnorm, g_mlp_up, MLP_HIDDEN, N_EMBD, GRP);
        swiglu(g_mlp_gate, g_mlp_up, g_mlp_hidden, MLP_HIDDEN);
        matmul_bitnet_ternary(p_down_q, p_down_s,
                        g_mlp_hidden, g_mlp_out, N_EMBD, MLP_HIDDEN, GRP);
        for (int d = 0; d < N_EMBD; ++d) g_x[d] += g_mlp_out[d];
        if (g_debug_inference && pos == ctx_len - 1) {
            print_debug_activation("mlp", l, g_x, N_EMBD);
        }
        if (g_debug_inference) Serial.printf("[DBG]   layer %d/%d done\n", l + 1, N_LAYER);
        yield();
    }
    rms_norm(g_x, g_xnorm, ln_f_gamma, N_EMBD);
    // Match Python BitLinearInference for the tied language-model head.
    // Do not reuse tok_emb_q here: input embeddings and the output head use
    // different quantization schemes.
    matmul_bitnet_ternary(lm_head_weights, lm_head_scales,
                          g_xnorm, g_logits, model_vocab_size, N_EMBD, GRP);
    if (!g_debug_inference) Serial.print('.');
}

static bool print_token(int id) {
    bool has_newline = false;
    if (id < 0 || id >= (int)model_vocab_size) return false;
    uint32_t start = pgm_read_dword(&model_vocab_offsets[id]);
    uint32_t end   = pgm_read_dword(&model_vocab_offsets[id + 1]);
    for (uint32_t i = start; i < end; ++i) {
        const uint8_t byte = pgm_read_byte(&model_vocab_bytes[i]);
        if (byte == '\n') {
            Serial.print('\n');
            has_newline = true;
        } else if (byte >= 0x20 && byte != 0x7f) {
            // Do not send model-generated control bytes (notably 0x1d) to
            // idf.py monitor: it treats them as local keyboard shortcuts.
            // UTF-8 bytes are >= 0x80, so Chinese output is preserved.
            Serial.print((char)byte);
        }
    }
    return has_newline;
}

static bool token_has_newline(int id) {
    if (id < 0 || id >= (int)model_vocab_size) return false;
    uint32_t start = pgm_read_dword(&model_vocab_offsets[id]);
    uint32_t end   = pgm_read_dword(&model_vocab_offsets[id + 1]);
    for (uint32_t i = start; i < end; ++i) {
        if (pgm_read_byte(&model_vocab_bytes[i]) == '\n') return true;
    }
    return false;
}

static int g_debug_sampling_index = 0;

static void print_debug_top_logits() {
    int top_ids[5] = { -1, -1, -1, -1, -1 };
    float top_logits[5] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    for (int id = 0; id < (int)model_vocab_size; ++id) {
        const float logit = g_logits[id];
        for (int rank = 0; rank < 5; ++rank) {
            if (logit > top_logits[rank]) {
                for (int move = 4; move > rank; --move) {
                    top_logits[move] = top_logits[move - 1];
                    top_ids[move] = top_ids[move - 1];
                }
                top_logits[rank] = logit;
                top_ids[rank] = id;
                break;
            }
        }
    }
    Serial.print("[DBG] top5 logits:");
    for (int rank = 0; rank < 5; ++rank) {
        Serial.printf(" #%d=id%d:%.5f", rank + 1, top_ids[rank], top_logits[rank]);
    }
    Serial.println();
}

static int few_shot_len = 0;

static void ctx_push(uint16_t id) {
    if (ctx_len >= INFER_CTX) {
        int evict_idx = (few_shot_len < INFER_CTX - 2) ? few_shot_len : 0;
        int shift_count = (INFER_CTX - 1) - evict_idx;
        if (shift_count > 0) {
            memmove(ctx_ids + evict_idx, ctx_ids + evict_idx + 1, shift_count * sizeof(uint16_t));
        }
        ctx_len = INFER_CTX - 1;
        if (ctx_pos > evict_idx) ctx_pos = evict_idx;
    }
    ctx_ids[ctx_len++] = id;
}

static void ctx_push_str(const char* text) {
    // ByteLevel BPE starts with one token per input byte and repeatedly merges
    // the adjacent pair with the smallest learned merge rank.  Do not replace
    // this with longest-string matching: it produces different Chinese IDs.
    uint16_t pieces[INFER_CTX * 2 + 16];
    int piece_count = 0;
    for (const uint8_t* p = (const uint8_t*)text; *p && piece_count < (int)(sizeof(pieces) / sizeof(pieces[0])); ++p) {
        pieces[piece_count++] = model_bpe_byte_tokens[*p];
    }

    while (piece_count > 1) {
        int best_index = -1;
        uint16_t best_rank = UINT16_MAX;
        uint16_t best_merged = 0;
        for (int i = 0; i + 1 < piece_count; ++i) {
            const uint32_t key = ((uint32_t)pieces[i] << 16) | pieces[i + 1];
            int low = 0;
            int high = (int)(sizeof(model_bpe_merges) / sizeof(model_bpe_merges[0])) - 1;
            while (low <= high) {
                const int mid = low + (high - low) / 2;
                const BpeMerge merge = model_bpe_merges[mid];
                if (merge.key < key) low = mid + 1;
                else if (merge.key > key) high = mid - 1;
                else {
                    if (merge.rank < best_rank) {
                        best_rank = merge.rank;
                        best_index = i;
                        best_merged = merge.merged;
                    }
                    break;
                }
            }
        }
        if (best_index < 0) break;
        pieces[best_index] = best_merged;
        memmove(&pieces[best_index + 1], &pieces[best_index + 2],
                (piece_count - best_index - 2) * sizeof(pieces[0]));
        --piece_count;
    }
    for (int i = 0; i < piece_count; ++i) {
        ctx_push(pieces[i]);
    }
}

static int sample_next(float temp) {
    if (ctx_len == 0) return 0;
    while (ctx_pos < ctx_len) {
        transformer_forward(ctx_ids[ctx_pos], ctx_pos);
        ctx_pos++;
        llm_optimistic_yield(1);
    }
    if (REPETITION_LOGIT_PENALTY > 0.0f) {
        int rep_window = (ctx_len < 20) ? ctx_len : 20;
        for (int c = ctx_len - rep_window; c < ctx_len; ++c) {
            uint16_t id = ctx_ids[c];
            if (id < model_vocab_size) {
                g_logits[id] -= REPETITION_LOGIT_PENALTY;
            }
        }
    }
    if (g_debug_inference) print_debug_top_logits();
    if (temp <= 0.0f) return argmax(g_logits, (int)model_vocab_size);
    for (int i = 0; i < (int)model_vocab_size; ++i) g_logits[i] /= temp;
    softmax(g_logits, (int)model_vocab_size);
#if defined(ESP8266) || defined(ESP8266_BOARD)
    float r = (float)random(0, 1000000) / 1000000.0f;
#else
    float r = (float)esp_random() / (float)UINT32_MAX;
#endif
    float cdf = 0.0f;
    for (int i = 0; i < (int)model_vocab_size; ++i) {
        cdf += g_logits[i];
        if (r <= cdf) return i;
    }
    return model_vocab_size - 1;
}

static const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    return p;
}

static bool parse_math_expr(const char*& p, double& val, int& op_count);

static bool parse_math_factor(const char*& p, double& val, int& op_count) {
    p = skip_ws(p);
    if (*p == '+') { p++; return parse_math_factor(p, val, op_count); }
    if (*p == '-') {
        p++;
        double sub = 0.0;
        if (!parse_math_factor(p, sub, op_count)) return false;
        val = -sub;
        return true;
    }
    if (*p == '(') {
        p++;
        if (!parse_math_expr(p, val, op_count)) return false;
        p = skip_ws(p);
        if (*p != ')') return false;
        p++;
        return true;
    }
    char* endp = nullptr;
    val = strtod(p, &endp);
    if (endp == p) return false;
    p = endp;

    p = skip_ws(p);
    if (*p == '^') {
        p++;
        op_count++;
        double exp = 0.0;
        if (!parse_math_factor(p, exp, op_count)) return false;
        val = pow(val, exp);
    }
    return true;
}

static bool parse_math_term(const char*& p, double& val, int& op_count) {
    if (!parse_math_factor(p, val, op_count)) return false;
    while (true) {
        p = skip_ws(p);
        char op = *p;
        if (op == '*' || op == '/' || op == '%') {
            p++;
            op_count++;
            double rhs = 0.0;
            if (!parse_math_factor(p, rhs, op_count)) return false;
            if (op == '*') val *= rhs;
            else if (op == '/') {
                if (fabs(rhs) < 1e-12) return false;
                val /= rhs;
            }
            else if (op == '%') {
                if (fabs(rhs) < 1e-12) return false;
                val = fmod(val, rhs);
            }
        } else {
            break;
        }
    }
    return true;
}

static bool parse_math_expr(const char*& p, double& val, int& op_count) {
    if (!parse_math_term(p, val, op_count)) return false;
    while (true) {
        p = skip_ws(p);
        char op = *p;
        if (op == '+' || op == '-') {
            p++;
            op_count++;
            double rhs = 0.0;
            if (!parse_math_term(p, rhs, op_count)) return false;
            if (op == '+') val += rhs;
            else val -= rhs;
        } else {
            break;
        }
    }
    return true;
}

static bool try_evaluate_math(const char* raw_input, char* output, size_t out_len) {
    if (!raw_input || !*raw_input) return false;

    char clean[128];
    size_t in_len = strlen(raw_input);
    if (in_len >= sizeof(clean)) in_len = sizeof(clean) - 1;
    memcpy(clean, raw_input, in_len);
    clean[in_len] = '\0';

    char lower[128];
    for (size_t i = 0; i <= in_len; ++i) {
        lower[i] = (char)tolower((unsigned char)clean[i]);
    }

    const char* start_ptr = clean;
    const char* lower_ptr = lower;

    const char* prefixes[] = {
        "what is", "what's", "whats", "calculate", "calc", "solve",
        "how much is", "evaluate", "compute", "tell me", "please calculate", "math:"
    };
    for (size_t k = 0; k < sizeof(prefixes)/sizeof(prefixes[0]); ++k) {
        const char* pfx = prefixes[k];
        size_t plen = strlen(pfx);
        if (strncmp(lower_ptr, pfx, plen) == 0) {
            start_ptr += plen;
            lower_ptr += plen;
            break;
        }
    }

    while (*start_ptr == ' ' || *start_ptr == ':' || *start_ptr == '\t') {
        start_ptr++;
    }

    char expr_buf[128];
    size_t elen = strlen(start_ptr);
    if (elen >= sizeof(expr_buf)) elen = sizeof(expr_buf) - 1;
    memcpy(expr_buf, start_ptr, elen);
    expr_buf[elen] = '\0';

    while (elen > 0 && (expr_buf[elen - 1] == '?' || expr_buf[elen - 1] == '=' ||
                        expr_buf[elen - 1] == ' ' || expr_buf[elen - 1] == '.' ||
                        expr_buf[elen - 1] == '!' || expr_buf[elen - 1] == '\r' ||
                        expr_buf[elen - 1] == '\n')) {
        expr_buf[--elen] = '\0';
    }

    if (elen == 0) return false;

    const char* p = expr_buf;
    double result = 0.0;
    int op_count = 0;
    if (!parse_math_expr(p, result, op_count)) return false;

    p = skip_ws(p);
    if (*p != '\0') return false; // Contains trailing non-math characters

    if (op_count == 0) return false; // Must contain at least one operator (+, -, *, /, %, ^)

    if (fabs(result - round(result)) < 1e-7 && fabs(result) < 1e14) {
        snprintf(output, out_len, "%lld", (long long)round(result));
    } else {
        snprintf(output, out_len, "%.6g", result);
    }
    return true;
}

static void clear_context() {
    ctx_len = 0;
    ctx_pos = 0;
}

// ASCII-only serial commands for validating UTF-8 Chinese prompts on terminals
// where an IME cannot send Chinese reliably. The answer is still generated by
// the model; it is intentionally not a hard-coded Q&A demo.
static const char* chinese_test_prompt(const char* command) {
    if (strcmp(command, ":zh1") == 0) return "嘟嘟可是谁的物品？";
    if (strcmp(command, ":dbgzh1") == 0) return "嘟嘟可是谁的物品？";
    if (strcmp(command, ":zh2") == 0) return "嘟嘟可这个名字是什么意思？";
    if (strcmp(command, ":zh3") == 0) return "嘟嘟可一族住在哪里？";
    if (strcmp(command, ":zh4") == 0) return "机械巨熊是基础角色设定吗？";
    if (strcmp(command, ":zh5") == 0) return "可莉会把嘟嘟可称作普通挂件吗？";
    if (strcmp(command, ":dbgzh5") == 0) return "可莉会把嘟嘟可称作普通挂件吗？";
    if (strcmp(command, ":zh6") == 0) return "嘟嘟可是由谁做出来送给可莉的？";
    if (strcmp(command, ":dbgzh6") == 0) return "嘟嘟可是由谁做出来送给可莉的？";
    if (strcmp(command, ":zh7") == 0) return "嘟嘟可一族远行时想寻找什么？";
    if (strcmp(command, ":dbgzh7") == 0) return "嘟嘟可一族远行时想寻找什么？";
    return nullptr;
}

static bool print_chinese_test_help(const char* command) {
    if (strcmp(command, ":zhhelp") != 0) return false;
    Serial.println("\nChinese model tests (real inference):");
    Serial.println("  :zh1  嘟嘟可是谁的物品？");
    Serial.println("  :dbgzh1  :zh1 with per-token/layer diagnostic logs");
    Serial.println("  :zh2  嘟嘟可这个名字是什么意思？");
    Serial.println("  :zh3  嘟嘟可一族住在哪里？");
    Serial.println("  :zh4  机械巨熊是基础角色设定吗？");
    Serial.println("  :zh5  可莉会把嘟嘟可称作普通挂件吗？");
    Serial.println("  :dbgzh5  :zh5 with per-token/layer diagnostic logs");
    Serial.println("  :zh6  嘟嘟可是由谁做出来送给可莉的？");
    Serial.println("  :dbgzh6  :zh6 with per-token/layer diagnostic logs");
    Serial.println("  :zh7  嘟嘟可一族远行时想寻找什么？");
    Serial.println("  :dbgzh7  :zh7 with per-token/layer diagnostic logs");
    return true;
}

static void run_chat() {
    Serial.print("\nUser: ");
    char input[INFER_CTX + 1];
    int  ilen = 0;
    while (true) {
        if (!Serial.available()) {
            delay(1);
            continue;
        }
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') { if (ilen > 0) break; continue; }
        if (c == '\b' || c == 0x7F) {
            if (ilen > 0) {
                ilen--;
                Serial.print("\b \b");
            }
            continue;
        }
        if (ilen < (int)sizeof(input) - 1) { input[ilen++] = c; Serial.print(c); }
    }
    input[ilen] = '\0';
    Serial.println();
    if (ilen == 0) return;

    g_debug_inference = false;

    if (print_chinese_test_help(input)) return;
    if (const char* preset = chinese_test_prompt(input)) {
        g_debug_inference = strncmp(input, ":dbgzh", 6) == 0;
        Serial.print("[Chinese test] ");
        Serial.println(preset);
        snprintf(input, sizeof(input), "%s", preset);
    }

    clear_context();

    char math_response[64];
    if (try_evaluate_math(input, math_response, sizeof(math_response))) {
        Serial.print("Bot: ");
        Serial.println(math_response);
        Serial.println("[0 ms - Math Harness]");
        
        char turn[INFER_CTX * 2];
        snprintf(turn, sizeof(turn), "User: %s\nBot: %s\n", input, math_response);
        ctx_push_str(turn);
        return;
    }

    // Preprocess input: lowercase, strip punctuation, fix "i'm"
    int len = strlen(input);
    while (len > 0 && (input[len-1] == '?' || input[len-1] == '!' || input[len-1] == '.' || input[len-1] == ' ')) {
        input[len-1] = '\0';
        len--;
    }
    for (int i = 0; i < len; ++i) {
        if (input[i] >= 'A' && input[i] <= 'Z') {
            input[i] = input[i] + 32;
        }
    }
    char clean_input[INFER_CTX * 2];
    if (strncmp(input, "i'm", 3) == 0) {
        snprintf(clean_input, sizeof(clean_input), "I'm%s", input + 3);
    } else if (strncmp(input, "i ", 2) == 0) {
        snprintf(clean_input, sizeof(clean_input), "I %s", input + 2);
    } else {
        snprintf(clean_input, sizeof(clean_input), "%s", input);
    }

    char prompt[INFER_CTX * 2 + 16];
    snprintf(prompt, sizeof(prompt), "User: %s\nBot:", clean_input);
    ctx_push_str(prompt);
    if (g_debug_inference) {
        Serial.printf("[DBG] prompt bytes=%u, encoded tokens=%d\n", (unsigned)strlen(prompt), ctx_len);
    }
    Serial.print("Bot:");
    if (!g_debug_inference) Serial.print(" [thinking...]");
    unsigned long t_start = millis();
    uint16_t output_ids[MAX_GEN_TOKENS];
    int output_len = 0;
    for (int i = 0; i < MAX_GEN_TOKENS; ++i) {
        llm_optimistic_yield(1);
        g_debug_sampling_index = i + 1;
        if (g_debug_inference) Serial.printf("\n[DBG] sampling output token %d\n", i + 1);
        int next_id  = sample_next(TEMPERATURE);
        if (g_debug_inference) Serial.printf("[DBG] sampled id=%d\n", next_id);
        if (next_id == 0) break; 
        bool hit_newline = token_has_newline(next_id);
        // Keep UTF-8 byte fragments together. Some Chinese characters span
        // BPE tokens, and USB Serial/JTAG is unreliable when they are emitted
        // one token at a time while inference continues.
        output_ids[output_len++] = (uint16_t)next_id;
        ctx_push((uint16_t)next_id);
        if (hit_newline && i > 1) break;
        llm_optimistic_yield(1);
    }
    if (ctx_len == 0 || ctx_ids[ctx_len - 1] != 199) { 
        ctx_push_str("\n");
    }
    unsigned long elapsed = millis() - t_start;
    if (g_debug_inference) {
        Serial.print("[DBG] generated token ids:");
        for (int i = 0; i < output_len; ++i) Serial.printf(" %u", output_ids[i]);
        Serial.print("\nBot (decoded):");
        for (int i = 0; i < output_len; ++i) print_token(output_ids[i]);
    } else {
        Serial.print("\nBot:");
        for (int i = 0; i < output_len; ++i) print_token(output_ids[i]);
    }
    Serial.println();
    Serial.printf("[%lu ms total]\n", elapsed);
}

static void print_model_info() {
    Serial.printf("  Vocab  : %u chars\n",   (unsigned)model_vocab_size);
    Serial.printf("  n_embd : %d\n",          N_EMBD);
    Serial.printf("  n_head : %d\n",          N_HEAD);
    Serial.printf("  n_layer: %d\n",          N_LAYER);
    Serial.printf("  ctx    : %d tokens\n",   INFER_CTX);
    Serial.printf("  mlp_h  : %d\n",          MLP_HIDDEN);
    Serial.printf("  flash  : %u bytes\n",    (unsigned)model_weights_len);
    Serial.printf("  HPP SHA-256: %s\n",      model_weights_hpp_sha256);
}

static bool alloc_buffers() {
    arena->reset();
    g_x          = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_kbuf       = (float*)arena->alloc(N_LAYER * INFER_CTX * N_KV_HEAD * HEAD_DIM * sizeof(float));
    g_vbuf       = (float*)arena->alloc(N_LAYER * INFER_CTX * N_KV_HEAD * HEAD_DIM * sizeof(float));
    g_xnorm      = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_qkv_out    = (float*)arena->alloc((N_EMBD + 2 * N_KV_HEAD * HEAD_DIM) * sizeof(float));
    g_attn_out   = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_proj_out   = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_att        = (float*)arena->alloc(INFER_CTX              * sizeof(float));
    g_mlp_gate   = (float*)arena->alloc(MLP_HIDDEN             * sizeof(float));
    g_mlp_up     = (float*)arena->alloc(MLP_HIDDEN             * sizeof(float));
    g_mlp_hidden = (float*)arena->alloc(MLP_HIDDEN             * sizeof(float));
    g_mlp_out    = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_logits     = (float*)arena->alloc(model_vocab_size       * sizeof(float));

#if defined(ESP32S3_BOARD) || defined(ESP_PLATFORM)
    int gate_n_groups = (N_EMBD + GRP - 1) / GRP;
    int gate_bytes_per_row = gate_n_groups * ((GRP + 4) / 5);
    size_t gate_q_sz = (size_t)MLP_HIDDEN * gate_bytes_per_row;
    size_t gate_s_sz = (size_t)MLP_HIDDEN * gate_n_groups * sizeof(uint16_t);

    int down_n_groups = (MLP_HIDDEN + GRP - 1) / GRP;
    int down_bytes_per_row = down_n_groups * ((GRP + 4) / 5);
    size_t down_q_sz = (size_t)N_EMBD * down_bytes_per_row;
    size_t down_s_sz = (size_t)N_EMBD * down_n_groups * sizeof(uint16_t);

    g_expert_cache.gate_q = (uint8_t*)arena->alloc(gate_q_sz);
    g_expert_cache.gate_s = (uint16_t*)arena->alloc(gate_s_sz);
    g_expert_cache.up_q   = (uint8_t*)arena->alloc(gate_q_sz);
    g_expert_cache.up_s   = (uint16_t*)arena->alloc(gate_s_sz);
    g_expert_cache.down_q = (uint8_t*)arena->alloc(down_q_sz);
    g_expert_cache.down_s = (uint16_t*)arena->alloc(down_s_sz);
    g_expert_cache.cached_layer = -1;
    g_expert_cache.cached_expert = -1;

    bool expert_ok = g_expert_cache.gate_q && g_expert_cache.gate_s &&
                     g_expert_cache.up_q && g_expert_cache.up_s &&
                     g_expert_cache.down_q && g_expert_cache.down_s;
#else
    bool expert_ok = true;
#endif

    return g_x && g_kbuf && g_vbuf && g_xnorm && g_qkv_out &&
           g_attn_out && g_proj_out && g_att && g_mlp_gate &&
           g_mlp_up && g_mlp_hidden && g_mlp_out && g_logits && expert_ok;
}

void setup() {
    Serial.begin(115200);
    delay(2000);

#if defined(ESP8266) || defined(ESP8266_BOARD)
    WiFi.mode(WIFI_OFF);
    WiFi.forceSleepBegin();
    delay(1);
    ESP.wdtEnable(5000);
    ESP.wdtFeed();
#elif defined(ESP32S3_BOARD) || defined(ESP_PLATFORM)
    if (!arena) {
        uint8_t* p = (uint8_t*)
#if defined(ESP_PLATFORM)
            heap_caps_malloc(ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#else
            ps_malloc(ARENA_SIZE);
#endif
        // 8MB octal PSRAM on N16R8
        if (!p) p = (uint8_t*)malloc(ARENA_SIZE);      // last-resort SRAM attempt
        if (p) arena = new MemoryArena(p, ARENA_SIZE);
    }
#else
    if (!arena) {
        arena = new MemoryArena(ARENA_SIZE);
    }
#endif

    Serial.println("\n╔══════════════════════════╗");
    Serial.println("║      ESP-LLM  v1.0       ║");
#if defined(ESP8266) || defined(ESP8266_BOARD)
    Serial.println("║ BitNet 1.58b on ESP8266  ║");
#elif defined(ESP32S3_BOARD) || defined(ESP_PLATFORM)
    Serial.println("║ BitNet 1.58b on ESP32-S3 ║");
#else
    Serial.println("║ BitNet 1.58b on ESP32    ║");
#endif
    Serial.println("╚══════════════════════════╝");
#if defined(ESP32S3_BOARD) || defined(ESP_PLATFORM)
    Serial.printf("PSRAM: %u B | Free heap at boot: %u B\n",
                  (unsigned)ESP.getPsramSize(), (unsigned)ESP.getFreeHeap());
#else
    Serial.printf("Free heap at boot (Wi-Fi OFF): %u B\n", (unsigned)ESP.getFreeHeap());
#endif

    if (!arena || !arena->is_valid() || !alloc_buffers()) {
        Serial.printf("[FAIL] Memory arena allocation failed (need %u KB, free %u B)\n",
                      (unsigned)(ARENA_SIZE / 1024), (unsigned)ESP.getFreeHeap());
        return;
    }
    Serial.printf("[OK] Memory Arena: %u KB (%u / %u B mapped)\n",
                  (unsigned)(ARENA_SIZE / 1024),
                  (unsigned)arena->used(), (unsigned)arena->capacity());
    print_model_info();
    Serial.printf("Free heap remaining: %u B\n\n", (unsigned)ESP.getFreeHeap());
    clear_context();
    Serial.println("Type your message and press Enter.");
    Serial.println("──────────────────────────────────");
}

void loop() {
    if (!arena || !arena->is_valid() || !g_x) {
        delay(5000);
        return;
    }
    run_chat();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) {
    setup();
    while (true) loop();
}
#endif
