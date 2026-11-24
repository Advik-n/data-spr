import unittest
from unittest.mock import Mock
from procurement.providers import OperationalError, Registry, free_chat


class ProvidersTest(unittest.TestCase):
    def test_baseline(self):
        r = Registry(env={})
        r.catalog = Mock(side_effect=AssertionError)
        self.assertFalse(any(r.models(p) for p in r.providers.values()))
        r.catalog.assert_not_called()

    def test_account_requires_confirmation(self):
        env = {"GROQ_API_KEY": "fake"}
        self.assertEqual(Registry("hosted", env).providers["groq"].status, "unconfirmed_billing")
        env["LLM_CONFIRMED_NO_CHARGE_PROVIDERS"] = "groq"
        self.assertEqual(Registry("hosted", env).providers["groq"].status, "eligible")
        self.assertEqual(Registry("local", env).providers["groq"].status, "profile_disabled")

    def test_free_catalog(self):
        model = {"architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                 "pricing": {"prompt": "0", "completion": "0"}}
        self.assertTrue(free_chat(model))
        self.assertFalse(free_chat(model, ["tools"]))
        self.assertFalse(free_chat(dict(model, pricing={"prompt": "0"})))
        model["pricing"]["request"] = "0.001"
        self.assertFalse(free_chat(model))

    def test_fallback_and_shared_budget(self):
        r = Registry("hosted", {"LLM_PROVIDER_ORDER": "custom", "CUSTOM_BASE_URL": "http://localhost",
            "CUSTOM_MODELS": "example", "LLM_CONFIRMED_NO_CHARGE_PROVIDERS": "custom",
            "LLM_MAX_POLICY_AUDIT_CALLS": "1"})
        class Fact:
            @classmethod
            def model_validate(cls, value):
                return value
        client = Mock()
        client.with_structured_output.side_effect = NotImplementedError
        message = client.bind_tools.return_value.invoke.return_value
        message.tool_calls = [{"name": "Fact", "args": {"amount": 1}}]
        message.response_metadata = {"model_name": "served"}
        r.client = Mock(return_value=client)
        value, meta = r.extract([], Fact, "row")
        self.assertEqual(value, {"amount": 1})
        self.assertEqual((meta["calls"], meta["method"], meta["serving_model"]), (2, "tool", "served"))
        r.reserve("row", True)
        with self.assertRaises(OperationalError):
            r.reserve("other", True)
        with self.assertRaises(OperationalError):
            r.reserve("row", False)
