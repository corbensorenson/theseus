// Windows-native PufferLib-compatible CPU backend for CartPole admission.
//
// This is not a template body or benchmark answer. It is a portability shim for
// PufferLib 4 on Windows: expose the `_C` surface needed by the PyTorch backend
// and Theseus RL lane while keeping the environment native, pointer-backed, and
// deterministic.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

constexpr float X_THRESHOLD = 2.4f;
constexpr float THETA_THRESHOLD_RADIANS = 12.0f * 2.0f * 3.14159265358979323846f / 360.0f;
constexpr int MAX_STEPS = 200;

double dict_get(py::dict dict, const char* key, double fallback) {
    if (!dict || !dict.contains(key)) {
        return fallback;
    }
    return py::cast<double>(dict[key]);
}

struct CartpoleEnv {
    float x = 0.0f;
    float x_dot = 0.0f;
    float theta = 0.0f;
    float theta_dot = 0.0f;
    int tick = 0;
    float episode_return = 0.0f;
    float cart_mass = 1.0f;
    float pole_mass = 0.1f;
    float pole_length = 0.5f;
    float gravity = 9.8f;
    float force_mag = 10.0f;
    float tau = 0.02f;
    int continuous = 0;
    std::minstd_rand rng;
};

struct Log {
    float perf = 0.0f;
    float episode_length = 0.0f;
    float x_threshold_termination = 0.0f;
    float pole_angle_termination = 0.0f;
    float max_steps_termination = 0.0f;
    float n = 0.0f;
    float score = 0.0f;
};

class VecEnv {
public:
    explicit VecEnv(py::dict args, int gpu_arg) {
        (void)gpu_arg;
        py::dict vec = args["vec"].cast<py::dict>();
        py::dict env_kwargs = args["env"].cast<py::dict>();
        total_agents = static_cast<int>(dict_get(vec, "total_agents", 64.0));
        if (total_agents < 1) {
            throw std::runtime_error("total_agents must be positive");
        }
        envs.resize(static_cast<size_t>(total_agents));
        observations.resize(static_cast<size_t>(total_agents) * obs_size);
        actions.resize(static_cast<size_t>(total_agents) * num_atns);
        rewards.resize(static_cast<size_t>(total_agents));
        terminals.resize(static_cast<size_t>(total_agents));

        const float cart_mass = static_cast<float>(dict_get(env_kwargs, "cart_mass", 1.0));
        const float pole_mass = static_cast<float>(dict_get(env_kwargs, "pole_mass", 0.1));
        const float pole_length = static_cast<float>(dict_get(env_kwargs, "pole_length", 0.5));
        const float gravity = static_cast<float>(dict_get(env_kwargs, "gravity", 9.8));
        const float force_mag = static_cast<float>(dict_get(env_kwargs, "force_mag", 10.0));
        const float tau = static_cast<float>(dict_get(env_kwargs, "dt", 0.02));
        const int continuous = static_cast<int>(dict_get(env_kwargs, "continuous", 0.0));

        for (int i = 0; i < total_agents; ++i) {
            CartpoleEnv& env = envs[static_cast<size_t>(i)];
            env.cart_mass = cart_mass;
            env.pole_mass = pole_mass;
            env.pole_length = pole_length;
            env.gravity = gravity;
            env.force_mag = force_mag;
            env.tau = tau;
            env.continuous = continuous;
            env.rng.seed(static_cast<uint32_t>(17 + i * 9973));
        }
        reset();
    }

    void reset() {
        for (int i = 0; i < total_agents; ++i) {
            reset_one(i);
        }
        std::fill(rewards.begin(), rewards.end(), 0.0f);
        std::fill(terminals.begin(), terminals.end(), 0.0f);
    }

    void cpu_step(std::uintptr_t actions_ptr) {
        const float* src = reinterpret_cast<const float*>(actions_ptr);
        if (src != nullptr) {
            std::memcpy(actions.data(), src, actions.size() * sizeof(float));
        }
        for (int i = 0; i < total_agents; ++i) {
            step_one(i);
        }
    }

    py::dict log() {
        py::dict out;
        if (log_acc.n <= 0.0f) {
            return out;
        }
        out["perf"] = log_acc.perf / log_acc.n;
        out["episode_length"] = log_acc.episode_length / log_acc.n;
        out["x_threshold_termination"] = log_acc.x_threshold_termination / log_acc.n;
        out["pole_angle_termination"] = log_acc.pole_angle_termination / log_acc.n;
        out["max_steps_termination"] = log_acc.max_steps_termination / log_acc.n;
        out["score"] = log_acc.score / log_acc.n;
        out["n"] = log_acc.n;
        log_acc = Log{};
        return out;
    }

    py::dict eval_log() {
        py::dict out;
        if (log_acc.n <= 0.0f) {
            return out;
        }
        out["perf"] = log_acc.perf / log_acc.n;
        out["episode_length"] = log_acc.episode_length / log_acc.n;
        out["score"] = log_acc.score / log_acc.n;
        out["n"] = log_acc.n;
        return out;
    }

    void close() {
        envs.clear();
        observations.clear();
        actions.clear();
        rewards.clear();
        terminals.clear();
    }

    void render(int env_id) {
        (void)env_id;
    }

    std::uintptr_t obs_ptr() { return reinterpret_cast<std::uintptr_t>(observations.data()); }
    std::uintptr_t actions_ptr() { return reinterpret_cast<std::uintptr_t>(actions.data()); }
    std::uintptr_t rewards_ptr() { return reinterpret_cast<std::uintptr_t>(rewards.data()); }
    std::uintptr_t terminals_ptr() { return reinterpret_cast<std::uintptr_t>(terminals.data()); }

    int total_agents = 0;
    int obs_size = 4;
    int num_atns = 1;
    std::vector<int> act_sizes{2};
    std::string obs_dtype = "FloatTensor";
    size_t obs_elem_size = sizeof(float);

private:
    std::vector<CartpoleEnv> envs;
    std::vector<float> observations;
    std::vector<float> actions;
    std::vector<float> rewards;
    std::vector<float> terminals;
    Log log_acc;

    float uniform_small(CartpoleEnv& env) {
        std::uniform_real_distribution<float> dist(-0.04f, 0.04f);
        return dist(env.rng);
    }

    void write_obs(int i) {
        CartpoleEnv& env = envs[static_cast<size_t>(i)];
        const size_t base = static_cast<size_t>(i) * obs_size;
        observations[base + 0] = env.x;
        observations[base + 1] = env.x_dot;
        observations[base + 2] = env.theta;
        observations[base + 3] = env.theta_dot;
    }

    void reset_one(int i) {
        CartpoleEnv& env = envs[static_cast<size_t>(i)];
        env.episode_return = 0.0f;
        env.x = uniform_small(env);
        env.x_dot = uniform_small(env);
        env.theta = uniform_small(env);
        env.theta_dot = uniform_small(env);
        env.tick = 0;
        write_obs(i);
    }

    void add_log(const CartpoleEnv& env) {
        log_acc.perf += env.episode_return > 0.0f ? env.episode_return / static_cast<float>(MAX_STEPS) : 0.0f;
        log_acc.episode_length += static_cast<float>(env.tick);
        log_acc.score += static_cast<float>(env.tick);
        log_acc.x_threshold_termination += (env.x < -X_THRESHOLD || env.x > X_THRESHOLD) ? 1.0f : 0.0f;
        log_acc.pole_angle_termination += (env.theta < -THETA_THRESHOLD_RADIANS || env.theta > THETA_THRESHOLD_RADIANS) ? 1.0f : 0.0f;
        log_acc.max_steps_termination += env.tick >= MAX_STEPS ? 1.0f : 0.0f;
        log_acc.n += 1.0f;
    }

    void step_one(int i) {
        CartpoleEnv& env = envs[static_cast<size_t>(i)];
        float a = actions[static_cast<size_t>(i)];
        if (!std::isfinite(a)) {
            a = 0.0f;
        }
        a = std::clamp(a, -1.0f, 1.0f);
        actions[static_cast<size_t>(i)] = a;

        const float force = env.continuous ? a * env.force_mag : (a > 0.5f ? env.force_mag : -env.force_mag);
        const float costheta = std::cos(env.theta);
        const float sintheta = std::sin(env.theta);
        const float total_mass = env.cart_mass + env.pole_mass;
        const float polemass_length = total_mass + env.pole_mass;
        const float temp = (force + polemass_length * env.theta_dot * env.theta_dot * sintheta) / total_mass;
        const float thetaacc = (env.gravity * sintheta - costheta * temp) /
            (env.pole_length * (4.0f / 3.0f - total_mass * costheta * costheta / total_mass));
        const float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        env.x += env.tau * env.x_dot;
        env.x_dot += env.tau * xacc;
        env.theta += env.tau * env.theta_dot;
        env.theta_dot += env.tau * thetaacc;
        env.tick += 1;

        const bool terminated = env.x < -X_THRESHOLD || env.x > X_THRESHOLD ||
            env.theta < -THETA_THRESHOLD_RADIANS || env.theta > THETA_THRESHOLD_RADIANS;
        const bool truncated = env.tick >= MAX_STEPS;
        const bool done = terminated || truncated;
        rewards[static_cast<size_t>(i)] = done ? 0.0f : 1.0f;
        env.episode_return += rewards[static_cast<size_t>(i)];
        terminals[static_cast<size_t>(i)] = terminated ? 1.0f : 0.0f;

        if (done) {
            add_log(env);
            reset_one(i);
        }
        write_obs(i);
    }
};

void puff_advantage_cpu(
        std::uintptr_t values_ptr, std::uintptr_t rewards_ptr,
        std::uintptr_t dones_ptr, std::uintptr_t importance_ptr,
        std::uintptr_t advantages_ptr,
        int num_steps, int horizon,
        float gamma, float lambda, float rho_clip, float c_clip) {
    const float* values = reinterpret_cast<const float*>(values_ptr);
    const float* rewards = reinterpret_cast<const float*>(rewards_ptr);
    const float* dones = reinterpret_cast<const float*>(dones_ptr);
    const float* importance = reinterpret_cast<const float*>(importance_ptr);
    float* advantages = reinterpret_cast<float*>(advantages_ptr);
    for (int row = 0; row < num_steps; row++) {
        int off = row * horizon;
        float lastpufferlam = 0.0f;
        for (int t = horizon - 2; t >= 0; t--) {
            int t_next = t + 1;
            float nextnonterminal = 1.0f - dones[off + t_next];
            float imp = importance[off + t];
            float rho_t = imp < rho_clip ? imp : rho_clip;
            float c_t = imp < c_clip ? imp : c_clip;
            float delta = rho_t * rewards[off + t_next] + gamma * values[off + t_next] * nextnonterminal - values[off + t];
            lastpufferlam = delta + gamma * lambda * c_t * lastpufferlam * nextnonterminal;
            advantages[off + t] = lastpufferlam;
        }
    }
}

std::unique_ptr<VecEnv> create_vec(py::dict args, int gpu = 0) {
    return std::make_unique<VecEnv>(args, gpu);
}

}  // namespace

PYBIND11_MODULE(_C, m) {
    m.attr("precision_bytes") = 4;
    m.attr("env_name") = "cartpole";
    m.attr("gpu") = 0;

    m.def("puff_advantage_cpu", &puff_advantage_cpu);
    m.def("create_vec", &create_vec, py::arg("args"), py::arg("gpu") = 0);

    py::class_<VecEnv, std::unique_ptr<VecEnv>>(m, "VecEnv")
        .def_readonly("total_agents", &VecEnv::total_agents)
        .def_readonly("obs_size", &VecEnv::obs_size)
        .def_readonly("num_atns", &VecEnv::num_atns)
        .def_readonly("act_sizes", &VecEnv::act_sizes)
        .def_readonly("obs_dtype", &VecEnv::obs_dtype)
        .def_readonly("obs_elem_size", &VecEnv::obs_elem_size)
        .def_property_readonly("gpu", [](VecEnv&) { return 0; })
        .def_property_readonly("obs_ptr", &VecEnv::obs_ptr)
        .def_property_readonly("actions_ptr", &VecEnv::actions_ptr)
        .def_property_readonly("rewards_ptr", &VecEnv::rewards_ptr)
        .def_property_readonly("terminals_ptr", &VecEnv::terminals_ptr)
        .def("reset", &VecEnv::reset)
        .def("cpu_step", &VecEnv::cpu_step)
        .def("render", &VecEnv::render)
        .def("log", &VecEnv::log)
        .def("eval_log", &VecEnv::eval_log)
        .def("close", &VecEnv::close);
}
