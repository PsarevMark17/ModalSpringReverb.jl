module ModalSpringReverb

import LinearAlgebra
import DSP
import LaTeXStrings
import Plots

function springParameters(
    HL::Float64, # Длина пружины (м)
    HD::Float64, # Диаметр пружины (м)
    NT::Int64, # Количество витков 
    WD::Float64 # Димаетр проволоки (м)
)::NTuple{4,Float64}

    R = HD / 2 # (Раздел 2.1)
    r = WD / 2 # (Раздел 2.1)
    α = rad2deg(atan(HL / (π * HD * NT))) # (Раздел 2.1)
    L = sqrt((π * HD * NT)^2 + HL^2) # (Раздел 2.1)

    return R, r, α, L

end

function nondimensionalization(
    E::Float64, # Модуль Юнга (Па)
    G::Float64, # Модуль сдвига (Па)
    ρ::Float64, # Плотность (кг/м^3)
    R::Float64, # Радиус пружины (м)
    r::Float64, # Радиус проволоки (м)
    α::Float64, # Угол наклона пружины (°)
    L::Float64, # Длина размотанной пружины (м)
    σ0::Float64, # Постоянное затухание (Гц)
    σ2::Float64, # Квадратичное затухания (с)
    φE::Float64, # Баланс поступательного и крутящего моментов источника (°)
    φP::Float64, # Баланс поступательного и крутящего моментов приёмника (°)
    M::Int64, # Количество пространственных разбиений
    fs::Int64 # Частота дискретизации (Гц)
)::NTuple{9,Float64}

    α = deg2rad(α) # ° → рад

    κ = cos(α)^2 / R # (7)
    A = π * r^2 # (8)
    J = π * r^4 / 4 # (9)
    Jφ = 2J # (10)

    s0 = 1 / κ # (14)
    t0 = 1 / (κ^2) * sqrt((ρ * A) / (E * J)) # (14)

    μ = tan(α) # (26)
    b = (E * J) / (G * Jφ) # (26)
    λ = L / s0 # (26)

    σ0 *= t0 # (119)
    σ2 /= t0 # (119)

    φE = deg2rad(φE) # ° → рад
    φP = deg2rad(φP) # ° → рад

    fs *= t0 # (120)

    Δs = λ / M # (63)
    Δt = 1 / fs # (121)

    return t0, μ, b, σ0, σ2, φE, φP, Δs, Δt

end

function dVectors(
    K::Int64 # Количество узлов дифференцирования с каждой стороны от центрального узла
)::Tuple{
    Vector{Float64},Vector{Float64}
}

    k = 1:K # Диапазон значений k

    a1 = @. ((-1)^(k + 1) * factorial(big(K))^2) / (k * factorial(big(K - k)) * factorial(big(K + k))) # (67)
    a2 = @. 2 * a1 / k # (68)

    d1 = a1 # (69) → (80)
    d2 = [a2[end:-1:1]; -2sum(a2); a2] # (70)

    return Float64.(d1), Float64.(d2)

end

function D2Matrix(
    M::Int64, # Количество пространственных разбиений
    d2::Vector{Float64} # Коэффициенты конечных разностей для второй производной
)::LinearAlgebra.Symmetric{Float64,Matrix{Float64}}

    K = trunc(Int, (size(d2, 1) - 1) / 2) # Количество узлов дифференцирования с каждой стороны от центрального узла

    D2 = LinearAlgebra.diagm(M - 1, M + 2 * K - 1, [i - 1 => d2[i] * ones(M - 1) for i in 1:2*K+1]...) # (87)

    D2[:, K+1:2*K-1] -= D2[:, K-1:-1:1] # (89)
    D2[:, end-2*K+2:end-K] -= D2[:, end:-1:end-K+2] # (89)

    return LinearAlgebra.Symmetric(D2[:, K+1:end-K])

end

function ZMatrix(
    μ::Float64, # Безразмерный параметр для кривизны
    b::Float64, # Безразмерный параметр для свойств пружины
    D2::LinearAlgebra.Symmetric{Float64,Matrix{Float64}} # Матричный разностный оператор 
)::Tuple{
    LinearAlgebra.Symmetric{Float64,Matrix{Float64}},Matrix{Float64}
}

    C1 = inv(LinearAlgebra.I - D2) # (97)
    C2 = inv(b * LinearAlgebra.I - D2) # (98)
    C3 = (1 - μ^2) * LinearAlgebra.I + D2 # (99)
    C4 = LinearAlgebra.I + D2 # (100)

    Z1 = D2 * (C2 * C3^2 + 4 * μ^2 * LinearAlgebra.I) # (93)
    Z2 = 2 * μ * D2 * C3 * (C2 * C4 - LinearAlgebra.I) # (94)
    Z3 = C1 * Z2 # (95)
    Z4 = D2 * C1 * (4 * μ^2 * C2 * C4^2 + C3^2) # (96)

    Z = [Z1 Z2; Z3 Z4] # (92)

    return C1, Z

end

function cVector(
    M::Int64, # Количество пространственных разбиений
    μ::Float64, # Безразмерный параметр для кривизны
    φE::Float64, # Баланс поступательного и крутящего моментов источника
    φP::Float64, # Баланс поступательного и крутящего моментов приёмника
    Δs::Float64, # Пространственный шаг
    d1::Vector{Float64}, # Коэффициенты конечных разностей для первой производной
    C1::LinearAlgebra.Symmetric{Float64,Matrix{Float64}}, # Матрица C1 из (97)
    q::Vector{Float64}, # Вектор собственных значений
    P::Matrix{Float64} # Матрица собственных векторов
)::Vector{Float64}

    ζE = [d1; zeros(M - size(d1, 1) - 1)] # (80)
    ζP = -ζE[end:-1:1] # (107)

    hE = [sin(φE) * ζE; (μ * sin(φE) - cos(φE)) * C1 * ζE] # (92) → (101) и (102)
    hP = Δs * [-sin(φP) * ζP; (cos(φP) - μ * sin(φP)) * ζP] # (104) → (105) и (106)

    cE = LinearAlgebra.inv(P) * hE # (110)
    cP = q .* (P') * hP # (114)

    c = cE .* cP # (117)

    return c

end

function abVectors(
    σ0::Float64, # Безразмерный параметр постоянного затухания
    σ2::Float64, # Безразмерный параметр квадратичного затухания
    Δt::Float64, # Временной шаг
    q::Vector{Float64} # Вектор собственных значений
)::Tuple{
    Vector{Float64},Vector{Float64}
}

    ω = sqrt.(-q) # (115)
    σ = σ2 * ω .^ 2 .+ σ0 # (118)
    ϵ = exp.(-σ * Δt) # (126)

    a = 2 * ϵ .* cos.(ω * Δt) #(124)
    b = -ϵ .^ 2 # (125)

    return a, b

end

function abcVectorsNondimension(
    M::Int64, # Количество пространственных разбиений
    K::Int64, # Количество узлов дифференцирования с каждой стороны от центрального узла
    t0::Float64, # Обезразмеривание времени (c)
    μ::Float64, # Безразмерный параметр для кривизны
    b::Float64, # Безразмерный параметр для свойств пружины
    σ0::Float64, # Безразмерный параметр постоянного затухания
    σ2::Float64, # Безразмерный параметр квадратичного затухания
    φE::Float64, # Баланс поступательного и крутящего моментов источника
    φP::Float64, # Баланс поступательного и крутящего моментов приёмника
    Δs::Float64, # Пространственный шаг
    Δt::Float64 # Временной шаг
)::Tuple{
    Vector{Float64},Vector{Float64},Vector{Float64}
}

    d1, d2 = dVectors(K) ./ Δs^2 # (Раздел 2.8)

    C1, Z = ZMatrix(μ, b, D2Matrix(M, d2)) # (Раздел 2.12)

    q, P = LinearAlgebra.eigen(Z) # (108)

    i = findall(qi -> (qi < 0) && sqrt(-qi) / (2 * π * t0) <= 20000, q) # (Раздел 2.15)

    println("Отрицательные собственные значения: ", length(findall(qi -> (qi > 0), q)))
    println("Моды вне слышимой части спектра: ", length(findall(qi -> (sqrt(abs(qi)) / (2 * π * t0) > 20000), q)))

    c = cVector(M, μ, φE, φP, Δs, d1, C1, q, P)[i] # (Раздел 2.15)

    a, b = abVectors(σ0, σ2, Δt, q[i]) # (Раздел 2.17)

    return a, b, c

end

abcVectors(
    E::Float64, # Модуль Юнга (Па)
    G::Float64, # Модуль сдвига (Па)
    ρ::Float64, # Плотность (кг/м^3)
    R::Float64, # Радиус пружины (м)
    r::Float64, # Радиус проволоки (м)
    α::Float64, # Угол наклона пружины (°)
    L::Float64, # Длина размотанной пружины (м)
    σ0::Float64, # Постоянное затухание (Гц)
    σ2::Float64, # Квадратичное затухания (с)
    φE::Float64, # Баланс поступательного и крутящего моментов источника (°)
    φP::Float64, # Баланс поступательного и крутящего моментов приёмника (°)
    M::Int64, # Количество пространственных разбиений
    fs::Int64, # Частота дискретизации (Гц)
    K::Int64 # Количество узлов дифференцирования с каждой стороны от центрального узла
)::Tuple{
    Vector{Float64},Vector{Float64},Vector{Float64}
} = abcVectorsNondimension(M, K, nondimensionalization(E, G, ρ, R, r, α, L, σ0, σ2, φE, φP, M, fs)...)

function unitImpulseResponse(
    fs::Int64, # Частота дискретизации (Гц)
    a::Vector{Float64}, # Вектор a из (124)
    b::Vector{Float64}, # Вектор b из (125)
    c::Vector{Float64}, # Вектор c из (117)
)::Vector{Float64}

    m = size(a, 1) # Количество мод
    n = ceil(Int, fs * 3) # Количество расчётных точек для 3 секунд симуляции

    yp = zeros(m) # Состояние на предыдущем шаге
    yc = zeros(m) # Состояние на текущем шаге
    yn = zeros(m) # Состояние на следующем шаге

    TP = zeros(n) # Показания приёмника
    TE = [ones(1); zeros(n - 1)] # Единичный импульс в источнике

    for t = 1:n
        yn = a .* yc + b .* yp + c * TE[t] # (123)
        yp, yc = yc, yn # Обновление шагов
        TP[t] = sum(yc) # (127)
    end

    return TP

end

function sonogram(
    NAME::String, # Название пружины
    F1::Float64, # Первая частота перехода
    F2::Float64, # Вторая частота перехода
    F3::Float64, # Третья частота перехода
    TP::Vector{Float64}, # Данные с приёмника
    fs::Int64, # Частота дискретизации
    n::Int64, # Количество сэмплов
    noverlap::Int64,  # Количество перекрытий
    nfft::Int64 # Количество узлов преобразования Фурье
)

    spec = DSP.Periodograms.spectrogram(TP, n, noverlap; fs=fs, nfft=nfft, window=DSP.Windows.blackman)
    Plots.heatmap(spec.time, spec.freq, DSP.Util.pow2db.(spec.power))
    Plots.hline!([20000], color=:orange, linewidth=2, label="20 кГц")
    Plots.hline!([F3], color=:cyan, linestyle=:dash, label=LaTeXStrings.L"f_3")
    Plots.hline!([F2], color=:darkcyan, linestyle=:dash, label=LaTeXStrings.L"f_2")
    Plots.hline!([F1], color=:blue, linestyle=:dash, label=LaTeXStrings.L"f_1")
    Plots.xlabel!("Время (с)")
    Plots.ylabel!("Частота (Гц)")
    Plots.title!(NAME)

end

precompile(springParameters, (Float64, Float64, Int64, Float64))
precompile(nondimensionalization, (Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Int64, Int64))
precompile(dVectors, (Int64,))
precompile(D2Matrix, (Int64, Vector{Float64}))
precompile(ZMatrix, (Float64, Float64, LinearAlgebra.Symmetric{Float64,Matrix{Float64}}))
precompile(cVector, (Int64, Float64, Float64, Float64, Float64, Vector{Float64}, LinearAlgebra.Symmetric{Float64,Matrix{Float64}}, Vector{Float64}, Matrix{Float64}))
precompile(abVectors, (Float64, Float64, Float64, Vector{Float64}))
precompile(abcVectorsNondimension, (Int64, Int64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64))
precompile(abcVectors, (Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Int64, Int64, Int64))
precompile(unitImpulseResponse, (Int64, Vector{Float64}, Vector{Float64}, Vector{Float64}))
precompile(sonogram, (String, Float64, Float64, Float64, Vector{Float64}, Int64, Int64, Int64, Int64))

export springParameters
export nondimensionalization
export abcVectorsNondimension
export abcVectors
export unitImpulseResponse
export sonogram

end
