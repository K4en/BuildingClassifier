Project Journal – Learning From Learning

This project started with a simple idea: build a model that can tell what kind of building it's looking at. But instead of relying on pre-trained networks or copying an existing approach, I wanted to see if I could create something from scratch — something that reflects how I think.

Along the way, I began to ask questions not about the data or the code, but about learning itself.

Why do we forget some things and hold on to others?
How do we decide what lessons matter?
What if a machine could do the same?

That's where the idea of a replay buffer came in — a way for the model to remember its most impactful examples from each training run. Not all the data. Just the moments that mattered. The ones that were either:

extremely confident (proof it “got it right”)

extremely wrong (proof it still had something to learn)

These are stored and replayed in future training sessions — like memory flashbacks. It's not about preventing forgetting. It's about remembering with purpose.

I didn't write the implementation code from scratch. I used ChatGPT to help me shape and scaffold it. But the thinking, the structure, and the design — that’s mine.

It might be chaotic. It might not be formal. But it's real. And that's worth documenting.

Appendix: The Original Thought That Sparked the Replay Buffer

I’m keeping this here because it’s how the idea first came to me. It wasn’t academic or clean. It was just me thinking aloud:

*“Well yeah. I mean I'm not a programming engineer, but I guess it should be possible to save some kind of reference point after each training along with something like I don't know... 10-20 datapoints that gave the most accuracy during training or helped the most or smth and when later updating the training use it as a sort of ‘yeah this IS old data, but it is a chair or house as well just as accurate than the new data’.

Or save the best and worst and add it again to the new training dataset.

I mean that's sort of how humans work too. Take what's working, keep updating, but sometimes despite all your new insights and evolved methods you still can't solve the problem and that's when you think back to the beginning of your journey and go like ‘waaaait a minute... I used to be able to solve this thing this way too, I know it's very archaic, but let's try it’ and boom, solved.”*

This wasn’t written for anyone but me, but I’m keeping it here to show that original ideas don’t always come in clean lines. Sometimes they come in half-sentences and “or smth” — and that’s okay.

This is where the replay buffer idea came from. This was the seed.
