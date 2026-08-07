# RESERVE - Revealed Preference over Stated Sentiment

**Status:** Measurement principle. Deliberately NOT a module.
**Urgency:** Applies immediately wherever recommendation quality is assessed.

## Principle
Weight behavioural acceptance and follow-through more heavily than passive sentiment.

Do not only measure what was recommended. Measure **what changed the person's next action.**

## Where it came from
A public thread where someone described feeling isolated. Commenters suggested volunteering,
libraries, class auditing, hobbies, age-appropriate social settings. No single answer
contained the solution. The signal emerged from aggregation plus noticing which suggestions
the person reacted strongly to - class auditing produced visible enthusiasm and an immediate
"I'm looking into that."

That reaction was more informative than ten upvotes on a different answer.

    stated problem -> candidate explanations -> multiple suggestions -> reaction
    -> revealed preference -> better next recommendation

## Application
    SoundKeep    a completed track outweighs a liked one; a crate built from a suggestion
                 outweighs a suggestion viewed. Sonic already learns from playback receipts
                 rather than ratings - this is the doctrine behind that choice.
    SURVIVOR     an acted-upon score outweighs a fetched one.
    any agent    the recommendation that changed the next action beats the one that was
                 rated highly and ignored.

## Why not a module
This belongs inside the receipt architecture already reserved, not beside it. Receipts
already record what happened after a decision; the principle is about which fields deserve
weight when evaluating quality.
