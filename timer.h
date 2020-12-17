#ifndef TIMER_H_
#define TIMER_H_

#include <iostream>
#include <chrono>       // timer.
#include <thread>

class Timer
{
public:
    /** \brief Constructor. */
    Timer () : start_time_ (std::chrono::high_resolution_clock::now())
    {
    }

    /** \brief Destructor. */
    virtual ~Timer () {}

    /** \brief Retrieve the time in microseconds spent since the last call to \a reset(). */
    inline double
    getTimeUs ()
    {
        std::chrono::system_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time_).count());
    }

    /** \brief Retrieve the time in milliseconds spent since the last call to \a reset(). */
    inline double
    getTime ()
    {
        std::chrono::system_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        return (std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time_).count());
    }

    /** \brief Retrieve the time in seconds spent since the last call to \a reset(). */
    inline double
    getTimeSeconds ()
    {
        return (getTime () * 0.001f);
    }

    // Sleep for a while.
    inline void
    sleepMilliSeconds(double mm_secs)
    {
        while(getTime() < mm_secs)
             std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    /** \brief Reset the Timer to 0. */
    inline void
    reset ()
    {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

protected:
//    boost::posix_time::ptime start_time_;
    std::chrono::system_clock::time_point start_time_;

};





#endif  // TIMER_H_
