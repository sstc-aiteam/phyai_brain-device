import { postJson } from "./core.js";

export function runFullDemo(data = {}, options = {})
{
	return postJson(
		"/api/demo/run_full_demo",
		data ?? {},
		{ signal: options.signal }
	);
}

export function runFullDemo2(options = {})
{
	return postJson(
		"/api/demo/run_full_demo_2",
		{},
		{ signal: options.signal }
	);
}

export function runDemoTopCabinet(options = {})
{
	return postJson(
		"/api/demo/run_demo_top_cabinet",
		{},
		{ signal: options.signal }
	);
}

export function runDemoSecondDrawer(options = {})
{
	return postJson(
		"/api/demo/run_demo_second_drawer",
		{},
		{ signal: options.signal }
	);
}

export function runDemoTrashCan(options = {})
{
	return postJson(
		"/api/demo/run_demo_trash_can",
		{},
		{ signal: options.signal }
	);
}


