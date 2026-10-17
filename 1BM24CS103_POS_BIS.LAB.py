import random

random.seed(42)

FEATURES = [
    "packet_rate",
    "failed_logins",
    "payload_size",
    "unusual_port",
    "connection_count"
]

POPULATION_SIZE = 40
MAX_GENERATIONS = 60
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8


def generate_traffic(samples=500):
    dataset = []

    for _ in range(samples):
        packet_rate = random.random()
        failed_logins = random.random()
        payload_size = random.random()
        unusual_port = random.randint(0, 1)
        connection_count = random.random()

        features = [
            packet_rate,
            failed_logins,
            payload_size,
            unusual_port,
            connection_count
        ]

        risk = (
            0.20 * packet_rate
            + 0.30 * failed_logins
            + 0.10 * payload_size
            + 0.25 * unusual_port
            + 0.15 * connection_count
        )

        risk += random.uniform(-0.12, 0.12)
        label = 1 if risk >= 0.48 else 0

        dataset.append((features, label))

    random.shuffle(dataset)
    return dataset


def create_individual():
    weights = [
        random.uniform(0.1, 1.0)
        for _ in FEATURES
    ]

    threshold = random.uniform(0.3, 0.7)
    return weights + [threshold]


def predict(individual, features):
    weights = individual[:-1]
    threshold = individual[-1]

    total_weight = sum(weights)

    if total_weight == 0:
        return 0

    score = sum(
        weight * feature
        for weight, feature in zip(weights, features)
    ) / total_weight

    return 1 if score >= threshold else 0


def calculate_fitness(individual, dataset):
    tp = fp = fn = 0

    for features, actual in dataset:
        predicted = predict(individual, features)

        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        elif predicted == 0 and actual == 1:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)


def select_parent(population, dataset):
    candidates = random.sample(
        population, min(3, len(population))
    )

    best = max(
        candidates,
        key=lambda rule: calculate_fitness(rule, dataset)
    )

    return best[:]


def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]

    point = random.randint(1, len(parent1) - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] += random.gauss(0, 0.12)

            if i == len(individual) - 1:
                individual[i] = max(
                    0.05, min(0.95, individual[i])
                )
            else:
                individual[i] = max(
                    0.01, min(1.0, individual[i])
                )

    return individual


def train_ga(training_data):
    population = [
        create_individual()
        for _ in range(POPULATION_SIZE)
    ]

    for generation in range(MAX_GENERATIONS):
        population.sort(
            key=lambda rule: calculate_fitness(
                rule, training_data
            ),
            reverse=True
        )

        best_rule = population[0][:]
        fitness = calculate_fitness(
            best_rule, training_data
        )

        if generation % 10 == 0:
            print(
                "Generation:", generation,
                "| Best Fitness:", round(fitness, 4)
            )

        new_population = [
            population[0][:],
            population[1][:]
        ]

        while len(new_population) < POPULATION_SIZE:
            parent1 = select_parent(population, training_data)
            parent2 = select_parent(population, training_data)

            child1, child2 = crossover(parent1, parent2)

            new_population.append(mutate(child1))

            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(child2))

        population = new_population

    population.sort(
        key=lambda rule: calculate_fitness(
            rule, training_data
        ),
        reverse=True
    )

    return population[0]


def evaluate(rule, test_data):
    tp = fp = tn = fn = 0

    for features, actual in test_data:
        predicted = predict(rule, features)

        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total * 100 if total else 0
    precision = tp / (tp + fp) * 100 if tp + fp else 0
    recall = tp / (tp + fn) * 100 if tp + fn else 0

    print("\n===== DETECTION RESULTS =====")
    print("Total test activities:", total)
    print("True attacks detected:", tp)
    print("False alarms:", fp)
    print("Normal activities correctly identified:", tn)
    print("Missed attacks:", fn)
    print("Accuracy: {:.2f}%".format(accuracy))
    print("Precision: {:.2f}%".format(precision))
    print("Recall: {:.2f}%".format(recall))


def monitor_network(rule, activities):
    print("\n===== NETWORK MONITORING =====")

    for number, features in enumerate(activities, start=1):
        predicted = predict(rule, features)

        print("\nActivity", number)

        for name, value in zip(FEATURES, features):
            print(name, ":", value)

        if predicted == 1:
            print("ALERT: Possible attack detected!")
            print("Action: Simulated blocking of activity")
        else:
            print("Status: Normal activity")
            print("Action: Allow normal network communication")


def main():
    print("ATTACK ALARM SYSTEM USING GENETIC ALGORITHM")

    data = generate_traffic(500)

    split = int(0.8 * len(data))
    training_data = data[:split]
    test_data = data[split:]

    print("\nTraining samples:", len(training_data))
    print("Testing samples:", len(test_data))

    print("\nTraining the Genetic Algorithm...")
    best_rule = train_ga(training_data)

    print("\n===== BEST DETECTION RULE =====")

    for name, weight in zip(FEATURES, best_rule[:-1]):
        print("{}: {:.3f}".format(name, weight))

    print("Decision threshold: {:.3f}".format(best_rule[-1]))

    print(
        "Training F1 score: {:.3f}".format(
            calculate_fitness(best_rule, training_data)
        )
    )

    evaluate(best_rule, test_data)

    sample_activities = [
        [0.10, 0.05, 0.20, 0, 0.10],
        [0.90, 0.95, 0.70, 1, 0.90],
        [0.30, 0.85, 0.40, 1, 0.65],
        [0.15, 0.10, 0.20, 0, 0.25]
    ]

    monitor_network(best_rule, sample_activities)

    print("\nSystem execution completed.")


if __name__ == "__main__":
    main()
