# # test requirement example: requirements_info(MCTSSolver(), UAVchasePOMDP(), statesA())

using POMDPs
using Random # for AbstractRNG
# using POMDPModelTools # for Deterministic
using Parameters
using StaticArrays
using Distributions
using LinearAlgebra
using Base
using POMCPOW
# importall POMDPs

const Vec2 = SVector{2, Float64}
const Vec3 = SVector{3, Float64}

struct states
    uavPose::Vec3
    uavHeading::Float64 # radius
    targetPose::Vec2
end

struct MDPAction
    xy_speed::Float64 # m/s
    z_speed::Float64
    angle::Float64 # radius
end

struct POMDPAction
    look::Bool
    vel_steer::MDPAction
end


@with_kw struct UAVchaseMDP <: MDP{states, MDPAction}
    # mu::Float64          = 2.0
    target_velocity::Vec2 = SVector(1, 0)
    target_std::Float64     = 0.05 # sensor noise
    dt::Float64          = 0.1
    target_height::Float64 = 0.3 # height of the platform
    landing_radius::Float64  = 0.1

    # physical constraints
    z_min::Float64 = 0.3
    z_max::Float64 = 5

    # might also consider collision cost
    r_detect::Float64 = 400.0
    r_outScene::Float64 = -200.0 # under the ground or above the ceiling
    r_action::Float64 = -20.0
    r_distance::Float64 = 100 # not sure how to relate states to this reward???

    # mission_terminate::Bool  = true
    discount::Float64    = 0.95 # when do I use discount?
end

@with_kw struct UAVchasePOMDP <: POMDP{states, MDPAction, Vec2}
    mdp::UAVchaseMDP           = UAVchaseMDP()
    meas_std::Float64          = 0.03 # camera measurement noise

    # angle to be within Field of View (FOV)
    FOV_angle1::Float64 = 60 # in degree
    FOV_angle2::Float64 = 60 # in degree
end

const UAVchaseProblem = Union{UAVchasePOMDP, UAVchaseMDP}
mdp(p::UAVchaseMDP) = p
mdp(p::UAVchasePOMDP) = p.mdp

POMDPs.discount(pp::UAVchasePOMDP) = mdp(pp).discount

# function next_target_pos(p::UAVchaseMDP, current_pos::Vec2)
#     dt_distance = p.dt*p.target_velocity
#     sensor_noise = rand(Normal(0, p.target_std), 2)
#     return current_pos + dt_distance + sensor_noise
# end

function POMDPs.generate_s(pp::UAVchaseProblem, s::states, a::MDPAction, rng::AbstractRNG)
    p = mdp(pp)
    # calculate target state
    target_dt_distance = p.dt*p.target_velocity
    sensor_noise = rand(Normal(0, p.target_std), 2)
    curr_targ = s.targetPose + target_dt_distance + sensor_noise # next_target_pos(p, s.targetPose)
    # calculate UAV state
    curr_angle = s.uavHeading + a.angle
    xy_dt_distance = p.dt*a.xy_speed*SVector(cos(curr_angle), sin(curr_angle)) # careful
    z_dt_distance = p.dt*a.z_speed
    xyz_dt_distance = SVector(xy_dt_distance[1], xy_dt_distance[2], z_dt_distance)
    curr_pos = s.uavPose + xyz_dt_distance
    return states(curr_pos, curr_angle, curr_targ)
end

function if_in_FOV(p::UAVchasePOMDP, s::states)
    # pose = s.uavPose
    range1 = s.uavPose[3]*tand(p.FOV_angle1)
    range2 = s.uavPose[3]*tand(p.FOV_angle2)
    # within_x = (s.targetPose[1]>(s.uavPose[1]-range1))||(s.targetPose[1]<(s.uavPose[1]+range1))
    if !(s.targetPose[1]>(s.uavPose[1]-range1))&&(s.targetPose[1]<(s.uavPose[1]+range1))
        return false # if target is NOT within FOV in x direction
    end
    if (s.targetPose[2]>(s.uavPose[2]-range2))&&(s.targetPose[2]<(s.uavPose[2]+range2))
        return true # if target is within FOV in y direction
    end
end
# I only count the case of POMDP, not having two reward functions as in VDPtag
function POMDPs.reward(pp::UAVchasePOMDP, a::MDPAction, sp::states)
    p = mdp(pp)
    action_reward = p.r_action * ((a.xy_speed != 0)||(a.z_speed != 0))
    detection_reward = p.r_detect * if_in_FOV(pp, sp)
    outScene_reward = p.r_outScene * ((sp.uavPose[3] < p.z_min)||(sp.uavPose[3] > p.z_max))
    # distance reward:
    target3Dpose = SVector(sp.targetPose[0], sp.targetPose[1], p.target_height)
    distance_reward = r_distance/norm(sp.uavPose - target3Dpose)
    return action_reward + detection_reward + outScene_reward + distance_reward
end

function POMDPs.generate_sr(pp::UAVchasePOMDP, s::states, a::MDPAction, rng::AbstractRNG)
    sp = generate_s(pp, s, a, rng)
    return sp, reward(pp,a, sp)
end


struct NullMeasurementNormal
    null::Bool
    mean::Vec2
    std::Float64  # not sure if I should specify covariance here? guess depends on how it will be used
    NullMeasurementNormal() = new(true)
    NullMeasurementNormal(mean::Vec2, std::Float64) = new(false, mean, std)
end

function observation(pp::UAVchasePOMDP, sp::states)
    if f_in_FOV(pp, sp)
        location = sp.targetPose # measurement is the true target location + noise
        return NullMeasurementNormal(location, pp.meas_std)
    else
        return NullMeasurementNormal()
    end
end

# assume having a measurment
function POMDPs.generate_o(pp::UAVchasePOMDP, s::states, a::MDPAction, sp::states, rng::AbstractRNG)
    if observation(pp, sp).null # if no measurement
        return s.targetPose # return last state
    else # if has measurement
        return sp.targetPose # return this state
    end
end

struct UAVInitDist end
POMDPs.sampletype(::Type{UAVInitDist}) = states
function rand(rng::AbstractRNG, d::UAVInitDist)
    return states(SVector(0.0, 0.0, 3.0), 0.0, 8.0*rand(rng, 2)-4.0)
end
POMDPs.initialstate_distribution(::UAVchaseProblem) = UAVInitDist()


# # struct LDNormalStateDist
# #     mean::Float64
# #     std::Float64
# # end
# #
# # sampletype(::Type{LDNormalStateDist}) = LightDark1DState
# # rand(rng::AbstractRNG, d::LDNormalStateDist) = LightDark1DState(0, d.mean + randn(rng)*d.std)
# # initialstate_distribution(pomdp::LightDark1D) = LDNormalStateDist(2, 3)

# # struct states
# #     uavPose::Vec3
# #     uavHeading::Float64 # radius
# #     targetPose::Vec2
# # end
