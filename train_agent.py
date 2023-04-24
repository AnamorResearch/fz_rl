import numpy as np

from rl_agent import RolloutBuffer


def pretrain(env, agent, pre_epochs, expert, max_step, gamma):
    expert_actions = expert['actions']
    expert_states = expert['states']

    for epoch in range(pre_epochs):
        exp_buffer = RolloutBuffer()

        for i in range(len(expert_actions)):
            exp_buffer.clear()
            actions = expert_actions[i][:, [1, 0]]
            states = expert_states[i]
            exp_buffer.actions = actions[:, :agent.action_space].reshape((-1, agent.action_space))
            exp_buffer.states = states
            exp_next_states = np.append(exp_buffer.states[1:], np.array([np.array([0] * agent.input_shape[0])]), axis=0)

            size = len(states)

            log_action_prob = agent.generator.feed_forward(states, exp_buffer.actions)
            exp_buffer.logprobs = log_action_prob

            rewards = np.array([env.reward(states[i][0], actions[i][0], actions[i][1]) for i in range(size)])
            exp_buffer.rewards = rewards
            exp_buffer.is_terminals = np.array([0 if i != size - 1 else 1 for i in range(size)])

            values = agent.generator.predict_v(exp_buffer.states)
            next_values = agent.generator.predict_v(exp_next_states)

            advantages = exp_buffer.cum_rewards(gamma, values, next_values)
            cum_rewards = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

            loss_list = agent.generator.pretrain_update(exp_buffer.states, exp_buffer.actions, cum_rewards, exp_buffer.logprobs, advantages, values)

        print("epoch:", epoch, ", pretrain_loss:", np.array(loss_list).mean())

        agent.buffer.clear()
        state = env.reset()
        for step in range(max_step):
            action, log_action_prob = agent.act(state=state)
            next_state, reward, done = env.step(*action)
            if step >= max_step - 1:
                done = True

            agent.buffer.push(action, state, next_state, log_action_prob, reward, done)

            state = next_state

            if done:
                break

        buffer = agent.buffer
        buffer.to_array()

        if epoch % 10 == 0:
            results = np.hstack((buffer.states, buffer.actions))
            np.savetxt('results/pre_result_' + str(epoch) + '.csv', results, delimiter=',')

    agent.generator.save_model(path="checkpoints/pretrain/model_P_gen")


def train_with_evaluate(env, feed, agent, epochs, max_step, gamma):
    lr_schedule_epopch = feed["lr_schedule_epopch"]
    gen_schedule = feed["gen_schedule"]
    stop_length = feed["stop_length"]

    rollout_buffer = []
    best_cost = 1e10

    for epoch in range(epochs):
        for idx, e in enumerate(lr_schedule_epopch):
            if e <= epoch:
                gen_lr = gen_schedule[idx]

        total_steps = 0

        early_stopping_count = 0
        try:
            if epoch:
                agent.generator.load_model(path="checkpoints/model_gen", noprint=True)
        except ValueError:
            pass

        agent.buffer.clear()
        state = env.reset()
        for step in range(max_step):
            action, log_action_prob = agent.act(state=state)
            next_state, reward, done = env.step(*action)
            if step >= max_step - 1:
                done = True

            agent.buffer.push(action, state, next_state, log_action_prob, reward, done)

            state = next_state
            total_steps += 1

            if done:
                break

        buffer = agent.buffer
        buffer.to_array()

        values = agent.generator.predict_v(buffer.states)
        next_values = agent.generator.predict_v(buffer.next_states)

        cost = (buffer.states[:, 0] - env.ideal_d[1:])**2 \
            + (buffer.actions[:, 0] - env.sample_v[1:])**2 \
            + (buffer.actions[:, 1] - env.sample_p[1:])**2

        advantages = buffer.cum_rewards(gamma, values, next_values)
        cum_rewards = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        _ = agent.generator.update(
            states=buffer.states, actions=buffer.actions,
            rewards=cum_rewards, log_action_probs=buffer.logprobs,
            advantages=advantages,
            v_old=values, lr=gen_lr)

        agent.generator.save_model(path="checkpoints/model_gen", noprint=True)
        early_stopping_count = 0
        better_cost = cost.mean()
        if better_cost < best_cost:
            best_cost = better_cost
            agent.generator.save_model(path="best_model/model_gen")
        if epoch % 10 == 0:
            results = np.hstack((buffer.states, buffer.actions))
            np.savetxt('results/result_' + str(epoch) + '.csv', results, delimiter=',')

        if early_stopping_count >= stop_length:
            print("stop")
            break

        print("loss: ", better_cost, " best: ", best_cost)

        all_buffer = RolloutBuffer()
        for i in rollout_buffer:
            all_buffer.actions.extend(list(i.actions[:-1]))
            all_buffer.states.extend(list(i.states[:-1]))
            all_buffer.next_states.extend(list(i.next_states[:-1]))
            all_buffer.logprobs.extend(list(i.logprobs[:-1]))
            all_buffer.rewards.extend(list(i.rewards[:-1]))
            all_buffer.costs.extend(list(i.costs[:-1]))
            all_buffer.is_terminals.extend(list(i.is_terminals[:-1]))

        all_buffer.to_array()

        print("====================")
        print('episode: %s' % (epoch + 1))
