#include "Tracker.h"
#include "cf.h"

/*#define BUILD_DLL
#ifdef BUILD_DLL
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif
extern "C" EXPORT void destory(void *tracker);*/
void destory(void *tracker)
{
	Tracker *tmp = (Tracker *)tracker;
	delete tmp;
}