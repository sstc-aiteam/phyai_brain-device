import { postJson } from "./core.js";

export function sendAI(text)
{
    return postJson("/api/agent/action",{user_text: text.trim()});
}

export function createSTT({ textOutputId, statusId, language = "zh-TW"} = {})
{
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const textOutput = document.getElementById(textOutputId);
    const statusElement = document.getElementById(statusId);

    let recognition = null;
    let isListening = false;
    let finalTranscript = "";
    let recognitionError = null;

    function emptyFunction()
    {
        return;
    }

    function setStatus(message)
    {
        if (statusElement)
        {
            statusElement.textContent = message;
        }
    }

    if (!textOutput || !statusElement)
    {
        console.error(
            "語音辨識初始化失敗：找不到指定的 HTML 元件",
            {
                textOutputId,
                statusId
            }
        );

        return {
            listening: emptyFunction,
            stop: emptyFunction
        };
    }

    if (!SpeechRecognition)
    {
        setStatus("目前瀏覽器不支援語音辨識功能");

        return {
            listening: emptyFunction,
            stop: emptyFunction
        };
    }

    recognition = new SpeechRecognition();

    recognition.lang = language;
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onstart = () =>
    {
        isListening = true;
        recognitionError = null;

        setStatus("正在聆聽，請開始說話");
    };

    recognition.onresult = event =>
    {
        let interimTranscript = "";

        for (let index = event.resultIndex; index < event.results.length; index++)
        {
            const result = event.results[index];
            const transcript = result[0].transcript;

            if (result.isFinal)
            {
                finalTranscript += transcript;
            }
            else
            {
                interimTranscript += transcript;
            }
        }

        textOutput.value = finalTranscript + interimTranscript;
    };

    recognition.onerror = event =>
    {
        recognitionError = event.error;

        console.error("語音辨識錯誤：", {
            error: event.error,
            message: event.message,
            event
        });

        if (event.error === "aborted")
        {
            return;
        }

        if (event.error === "no-speech")
        {
            setStatus("沒有偵測到語音");
            return;
        }

        setStatus(`語音辨識失敗：${event.error}`);
    };

    recognition.onend = () =>
    {
        isListening = false;

        if (!recognitionError)
        {
            setStatus("語音辨識已停止");
        }
    };

    function listening()
    {
        if (isListening)
        {
            stop();
            return;
        }
        
        textOutput.value = "";
        finalTranscript = "";
        recognitionError = null;

        try
        {
            recognition.start();
        }
        catch (error)
        {
            console.error(
                "無法啟動語音辨識：",
                error
            );

            setStatus(
                `無法啟動語音辨識：${error.message}`
            );
        }
    }

    function stop()
    {
        if (!recognition || !isListening)
        {
            return;
        }

        recognition.stop();
    }

    return {
        listening,
        stop
    };
}

export function createTTS({language = "zh-TW", rate = 1, pitch = 1, volume = 1, statusId = null} = {})
{
    const speechSynthesis = window.speechSynthesis;

    const statusElement = statusId ? document.getElementById(statusId) : null;

    let currentUtterance = null;
    let isSpeaking = false;

    function setStatus(message)
    {
        if (statusElement)
        {
            statusElement.textContent = message;
        }
    }

    function emptyFunction()
    {
        return null;
    }

    if (!("speechSynthesis" in window))
    {
        console.error("目前瀏覽器不支援 Web Speech API");

        setStatus("目前瀏覽器不支援語音播放功能");

        return {
            speak: emptyFunction,
            stop: emptyFunction,
            speaking: () => false
        };
    }

    function speak(content, options = {})
    {
        if (typeof content !== "string" || !content.trim())
        {
            console.error(
                "speak(content)：content 必須是非空字串"
            );

            setStatus("沒有可朗讀的文字");

            return null;
        }

        const resolvedLanguage = options.language ?? language;
        const resolvedRate = options.rate ?? rate;
        const resolvedPitch = options.pitch ?? pitch;
        const resolvedVolume = options.volume ?? volume;

        // 停止前一次尚未完成的語音
        speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(content.trim());

        utterance.lang = resolvedLanguage;
        utterance.rate = resolvedRate;
        utterance.pitch = resolvedPitch;
        utterance.volume = resolvedVolume;

        utterance.onstart = () =>
        {
            isSpeaking = true;
            currentUtterance = utterance;

            setStatus("正在語音播放");
        };

        utterance.onend = () =>
        {
            isSpeaking = false;

            if (currentUtterance === utterance)
            {
                currentUtterance = null;
            }

            setStatus("語音播放完成");
        };

        utterance.onerror = event =>
        {
            isSpeaking = false;

            if (currentUtterance === utterance)
            {
                currentUtterance = null;
            }

            if (
                event.error === "canceled" ||
                event.error === "interrupted"
            )
            {
                return;
            }

            console.error(
                "語音播放失敗：",
                {
                    error: event.error,
                    event
                }
            );

            setStatus(
                `語音播放失敗：${event.error}`
            );
        };

        currentUtterance = utterance;

        speechSynthesis.speak(utterance);

        return utterance;
    }

    function stop()
    {
        if (!isSpeaking && !speechSynthesis.speaking)
        {
            return;
        }

        speechSynthesis.cancel();

        isSpeaking = false;
        currentUtterance = null;

        setStatus("語音播放已停止");
    }

    function speaking()
    {
        return (
            isSpeaking ||
            speechSynthesis.speaking
        );
    }

    return {
        speak,
        stop,
        speaking
    };
}