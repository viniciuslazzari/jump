using JuMP
using HiGHS

struct Point
  x::Int32
  y::Int32
end

function read_file(path)
  f = open(path)

  k = parse(Int64, readline(f))
  arr = Point[]

  while !eof(f)
    s = readline(f)
    i, x, y = split(s, " ")
    push!(arr, Point(parse(Int32, x), parse(Int32, y)))
  end

  close(f)

  return k, arr, length(arr)
end

function euclidian_distance(x1::Int32, y1::Int32, x2::Int32, y2::Int32)
  return sqrt((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
end

function build_distance_matrix(arr::Vector{Point}, n::Int64)
  m = zeros(Float64, n, n)

  for i = 1:n
    for j = 1:n
      m[i, j] = euclidian_distance(arr[i].x, arr[i].y, arr[j].x, arr[j].y)
    end
  end

  return m
end

function build_model(k::Int64, n::Int64, m::Matrix{Float64})
  model = Model(HiGHS.Optimizer)

  @variable(model, t_max >= 0)
  @variable(model, x[1:n, 1:n, 1:k], Bin)
  @variable(model, 2 <= u[1:n] <= n)

  @objective(model, Min, t_max)

  # Each point should be exited exactly once (excluding the depot)
  @constraint(model, [i in 2:n], sum(x[i, j, p] for p in 1:k, j in 1:n) == 1)

  # Each point should be entered exactly once (excluding the depot)
  @constraint(model, [j in 2:n], sum(x[i, j, p] for p in 1:k, i in 1:n) == 1)

  # t_max should be higher than all sum of route times
  @constraint(model, [p in 1:k], sum(x[i, j, p] * m[i, j] for i in 1:n, j in 1:n) <= t_max)

  # No point could be entered and exited at the same time (i.e., no self-loops)
  @constraint(model, [p in 1:k], sum(x[w, w, p] for w in 1:n) == 0)
  
  # The depot (C) should be exited only 1 time in each route
  @constraint(model, [p in 1:k], sum(x[1, j, p] for j in 2:n) == 1)

  # The depot (C) should be entered only 1 time in each route
  @constraint(model, [p in 1:k], sum(x[i, 1, p] for i in 2:n) == 1)

  # Conservation of flow: each point must be entered and exited the same number of times
  for p in 1:k
    @constraint(model, [w in 1:n], sum(x[i, w, p] for i in 1:n) == sum(x[w, j, p] for j in 1:n))
  end

  # MTZ subtour elimination constraints
  for p in 1:k
    for i in 2:n
      for j in 2:n
        if i != j
          @constraint(model, u[i] - u[j] + n * x[i, j, p] <= n - 1)
        end
      end
    end
  end

  return model, x
end

function main()
  file = ARGS[1]
  path = "instances/" * file

  k, arr, n = read_file(path)
  m = build_distance_matrix(arr, n)
  model, x = build_model(k, n, m)

  optimize!(model)
  println(objective_value(model))
end

main()